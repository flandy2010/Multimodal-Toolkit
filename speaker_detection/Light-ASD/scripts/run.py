"""
Light-ASD single video active speaker detection.

Usage:
    cd speaker_detection/light-ASD
    python scripts/run.py --video_path /path/to/video.mp4 \
                          --pretrain_model weight/pretrain_AVA_CVPR.model

Pipeline:
    1. ffmpeg 提取音频 + 视频帧 (25fps)
    2. 场景检测 (scenedetect)
    3. 人脸检测 (S3FD)
    4. 人脸追踪 + 裁剪人脸片段
    5. ASD 模型推理 → 每个 track 得到逐帧 speaking score
    6. 输出可视化视频 (标注说话人)

Dependencies: torch, opencv-python, scipy, python_speech_features, scenedetect, ffmpeg
"""

import sys
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import time
import argparse
import glob
import subprocess
import warnings
import math
import pickle

import cv2
import numpy
import torch
import python_speech_features
from scipy import signal
from scipy.io import wavfile
from scipy.interpolate import interp1d
from shutil import rmtree

from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.faceDetector.s3fd import S3FD
from ASD import ASD

warnings.filterwarnings("ignore")


def scene_detect(args):
    """Scene detection, returns list of shot time durations."""
    videoManager = VideoManager([args.videoFilePath])
    statsManager = StatsManager()
    sceneManager = SceneManager(statsManager)
    sceneManager.add_detector(ContentDetector())
    baseTimecode = videoManager.get_base_timecode()
    videoManager.set_downscale_factor()
    videoManager.start()
    sceneManager.detect_scenes(frame_source=videoManager)
    sceneList = sceneManager.get_scene_list(baseTimecode)
    if sceneList == []:
        sceneList = [(videoManager.get_base_timecode(), videoManager.get_current_timecode())]
    savePath = os.path.join(args.pyworkPath, 'scene.pckl')
    with open(savePath, 'wb') as fil:
        pickle.dump(sceneList, fil)
    print(f"[scene_detect] {len(sceneList)} scenes detected")
    return sceneList


def face_detect(args):
    """Face detection on all frames using S3FD."""
    DET = S3FD(device=args.device)
    flist = sorted(glob.glob(os.path.join(args.pyframesPath, '*.jpg')))
    dets = []
    for fidx, fname in enumerate(flist):
        image = cv2.imread(fname)
        imageNumpy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes = DET.detect_faces(imageNumpy, conf_th=0.9, scales=[args.facedetScale])
        dets.append([])
        for bbox in bboxes:
            dets[-1].append({'frame': fidx, 'bbox': (bbox[:-1]).tolist(), 'conf': bbox[-1]})
        sys.stderr.write(f'\r[face_detect] frame {fidx:05d}, {len(dets[-1])} faces')
    sys.stderr.write('\n')
    savePath = os.path.join(args.pyworkPath, 'faces.pckl')
    with open(savePath, 'wb') as fil:
        pickle.dump(dets, fil)
    return dets


def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou


def track_shot(args, sceneFaces):
    """Face tracking within a single shot."""
    iouThres = 0.5
    tracks = []
    while True:
        track = []
        for frameFaces in sceneFaces:
            for face in frameFaces:
                if track == []:
                    track.append(face)
                    frameFaces.remove(face)
                elif face['frame'] - track[-1]['frame'] <= args.numFailedDet:
                    iou = bb_intersection_over_union(face['bbox'], track[-1]['bbox'])
                    if iou > iouThres:
                        track.append(face)
                        frameFaces.remove(face)
                        continue
                else:
                    break
        if track == []:
            break
        elif len(track) > args.minTrack:
            frameNum = numpy.array([f['frame'] for f in track])
            bboxes = numpy.array([numpy.array(f['bbox']) for f in track])
            frameI = numpy.arange(frameNum[0], frameNum[-1] + 1)
            bboxesI = []
            for ij in range(0, 4):
                interpfn = interp1d(frameNum, bboxes[:, ij])
                bboxesI.append(interpfn(frameI))
            bboxesI = numpy.stack(bboxesI, axis=1)
            if max(numpy.mean(bboxesI[:, 2] - bboxesI[:, 0]),
                   numpy.mean(bboxesI[:, 3] - bboxesI[:, 1])) > args.minFaceSize:
                tracks.append({'frame': frameI, 'bbox': bboxesI})
    return tracks


def crop_video(args, track, cropFile):
    """Crop face clips (video + audio) for a single track."""
    flist = sorted(glob.glob(os.path.join(args.pyframesPath, '*.jpg')))
    vOut = cv2.VideoWriter(cropFile + 't.avi', cv2.VideoWriter_fourcc(*'XVID'), 25, (224, 224))
    dets = {'x': [], 'y': [], 's': []}
    for det in track['bbox']:
        dets['s'].append(max((det[3] - det[1]), (det[2] - det[0])) / 2)
        dets['y'].append((det[1] + det[3]) / 2)
        dets['x'].append((det[0] + det[2]) / 2)
    dets['s'] = signal.medfilt(dets['s'], kernel_size=13)
    dets['x'] = signal.medfilt(dets['x'], kernel_size=13)
    dets['y'] = signal.medfilt(dets['y'], kernel_size=13)
    for fidx, frame in enumerate(track['frame']):
        cs = args.cropScale
        bs = dets['s'][fidx]
        bsi = int(bs * (1 + 2 * cs))
        image = cv2.imread(flist[frame])
        frame_padded = numpy.pad(image, ((bsi, bsi), (bsi, bsi), (0, 0)), 'constant', constant_values=(110, 110))
        my = dets['y'][fidx] + bsi
        mx = dets['x'][fidx] + bsi
        face = frame_padded[int(my - bs):int(my + bs * (1 + 2 * cs)),
                            int(mx - bs * (1 + cs)):int(mx + bs * (1 + cs))]
        vOut.write(cv2.resize(face, (224, 224)))
    audioTmp = cropFile + '.wav'
    audioStart = (track['frame'][0]) / 25
    audioEnd = (track['frame'][-1] + 1) / 25
    vOut.release()
    command = ("ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -threads %d -ss %.3f -to %.3f %s -loglevel panic" %
               (args.audioFilePath, args.nDataLoaderThread, audioStart, audioEnd, audioTmp))
    subprocess.call(command, shell=True, stdout=None)
    command = ("ffmpeg -y -i %st.avi -i %s -threads %d -c:v copy -c:a copy %s.avi -loglevel panic" %
               (cropFile, audioTmp, args.nDataLoaderThread, cropFile))
    subprocess.call(command, shell=True, stdout=None)
    if os.path.exists(cropFile + 't.avi'):
        os.remove(cropFile + 't.avi')
    return {'track': track, 'proc_track': dets}


def evaluate_network(files, args):
    """ASD inference on cropped face clips."""
    s = ASD()
    s.loadParameters(args.pretrain_model, map_location=args.device)
    s.to(args.device)
    print(f"[ASD] Model loaded from {args.pretrain_model} on {args.device}")
    s.eval()
    allScores = []
    durationSet = {1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6}
    for file in files:
        fileName = os.path.splitext(file.split('/')[-1])[0]
        _, audio = wavfile.read(os.path.join(args.pycropPath, fileName + '.wav'))
        audioFeature = python_speech_features.mfcc(audio, 16000, numcep=13, winlen=0.025, winstep=0.010)
        video = cv2.VideoCapture(os.path.join(args.pycropPath, fileName + '.avi'))
        videoFeature = []
        while video.isOpened():
            ret, frames = video.read()
            if ret:
                face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
                face = cv2.resize(face, (224, 224))
                face = face[int(112 - 112 / 2):int(112 + 112 / 2), int(112 - 112 / 2):int(112 + 112 / 2)]
                videoFeature.append(face)
            else:
                break
        video.release()
        videoFeature = numpy.array(videoFeature)
        length = min((audioFeature.shape[0] - audioFeature.shape[0] % 4) / 100, videoFeature.shape[0] / 25)
        if length <= 0:
            allScores.append(numpy.array([0.0]))
            continue
        audioFeature = audioFeature[:int(round(length * 100)), :]
        videoFeature = videoFeature[:int(round(length * 25)), :, :]
        allScore = []
        for duration in durationSet:
            batchSize = int(math.ceil(length / duration))
            scores = []
            with torch.no_grad():
                for i in range(batchSize):
                    inputA = torch.FloatTensor(audioFeature[i * duration * 100:(i + 1) * duration * 100, :]).unsqueeze(0).to(args.device)
                    inputV = torch.FloatTensor(videoFeature[i * duration * 25:(i + 1) * duration * 25, :, :]).unsqueeze(0).to(args.device)
                    embedA = s.model.forward_audio_frontend(inputA)
                    embedV = s.model.forward_visual_frontend(inputV)
                    out = s.model.forward_audio_visual_backend(embedA, embedV)
                    score = s.lossAV.forward(out, labels=None)
                    scores.extend(score)
            allScore.append(scores)
        allScore = numpy.round((numpy.mean(numpy.array(allScore), axis=0)), 1).astype(float)
        allScores.append(allScore)
    return allScores


def visualization(tracks, scores, args, output_path=None):
    """Render output video with speaker detection boxes."""
    flist = sorted(glob.glob(os.path.join(args.pyframesPath, '*.jpg')))
    faces = [[] for _ in range(len(flist))]
    for tidx, track in enumerate(tracks):
        score = scores[tidx]
        for fidx, frame in enumerate(track['track']['frame'].tolist()):
            s = score[max(fidx - 2, 0): min(fidx + 3, len(score) - 1)]
            s = numpy.mean(s)
            faces[frame].append({
                'track': tidx, 'score': float(s),
                's': track['proc_track']['s'][fidx],
                'x': track['proc_track']['x'][fidx],
                'y': track['proc_track']['y'][fidx]
            })
    firstImage = cv2.imread(flist[0])
    fw = firstImage.shape[1]
    fh = firstImage.shape[0]
    video_only_path = os.path.join(args.pyaviPath, 'video_only.avi')
    vOut = cv2.VideoWriter(video_only_path, cv2.VideoWriter_fourcc(*'XVID'), 25, (fw, fh))
    for fidx, fname in enumerate(flist):
        image = cv2.imread(fname)
        for face in faces[fidx]:
            clr = 255 if face['score'] >= 0 else 0
            txt = round(face['score'], 1)
            cv2.rectangle(image,
                          (int(face['x'] - face['s']), int(face['y'] - face['s'])),
                          (int(face['x'] + face['s']), int(face['y'] + face['s'])),
                          (0, clr, 255 - clr), 3)
            cv2.putText(image, '%s' % txt,
                        (int(face['x'] - face['s']), int(face['y'] - face['s'])),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, clr, 255 - clr), 2)
        vOut.write(image)
    vOut.release()

    if output_path is None:
        output_path = os.path.join(args.pyaviPath, 'video_out.avi')
    command = ("ffmpeg -y -i %s -i %s -threads %d -c:v copy -c:a copy %s -loglevel panic" %
               (video_only_path, args.audioFilePath, args.nDataLoaderThread, output_path))
    subprocess.call(command, shell=True, stdout=None)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Light-ASD: Single Video Speaker Detection")
    parser.add_argument('--video_path', type=str, required=True, help='Path to input video')
    parser.add_argument('--pretrain_model', type=str, default='weight/pretrain_AVA_CVPR.model',
                        help='Path to pretrained ASD model')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Output video path (default: <save_dir>/pyavi/video_out.avi)')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory for intermediate results (default: same dir as video)')
    parser.add_argument('--facedet_scale', type=float, default=0.25,
                        help='Scale factor for face detection')
    parser.add_argument('--crop_scale', type=float, default=0.40,
                        help='Scale bounding box for face crop')
    parser.add_argument('--min_track', type=int, default=10,
                        help='Minimum frames for a valid track')
    parser.add_argument('--num_failed_det', type=int, default=10,
                        help='Allowed missed detections before stopping tracking')
    parser.add_argument('--min_face_size', type=int, default=1,
                        help='Minimum face size in pixels')
    parser.add_argument('--n_data_loader_thread', type=int, default=10,
                        help='Number of ffmpeg threads')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device for inference (cpu/cuda/cuda:0)')
    pargs = parser.parse_args()

    # Setup args namespace (compatible with Columbia_test.py functions)
    args = argparse.Namespace()
    args.videoPath = pargs.video_path
    args.pretrain_model = pargs.pretrain_model
    args.device = pargs.device
    args.facedetScale = pargs.facedet_scale
    args.cropScale = pargs.crop_scale
    args.minTrack = pargs.min_track
    args.numFailedDet = pargs.num_failed_det
    args.minFaceSize = pargs.min_face_size
    args.nDataLoaderThread = pargs.n_data_loader_thread

    # Determine save directory
    video_name = os.path.splitext(os.path.basename(args.videoPath))[0]
    if pargs.save_dir:
        args.savePath = pargs.save_dir
    else:
        args.savePath = os.path.join(os.path.dirname(args.videoPath), video_name + '_asd')

    # Create working directories
    args.pyaviPath = os.path.join(args.savePath, 'pyavi')
    args.pyframesPath = os.path.join(args.savePath, 'pyframes')
    args.pyworkPath = os.path.join(args.savePath, 'pywork')
    args.pycropPath = os.path.join(args.savePath, 'pycrop')

    if os.path.exists(args.savePath):
        rmtree(args.savePath)
    os.makedirs(args.pyaviPath, exist_ok=True)
    os.makedirs(args.pyframesPath, exist_ok=True)
    os.makedirs(args.pyworkPath, exist_ok=True)
    os.makedirs(args.pycropPath, exist_ok=True)

    print(f"[1/6] Extracting video and audio...")
    # Extract video at 25fps
    args.videoFilePath = os.path.join(args.pyaviPath, 'video.avi')
    command = ("ffmpeg -y -i %s -qscale:v 2 -threads %d -async 1 -r 25 %s -loglevel panic" %
               (args.videoPath, args.nDataLoaderThread, args.videoFilePath))
    subprocess.call(command, shell=True, stdout=None)

    # Extract audio
    args.audioFilePath = os.path.join(args.pyaviPath, 'audio.wav')
    command = ("ffmpeg -y -i %s -qscale:a 0 -ac 1 -vn -threads %d -ar 16000 %s -loglevel panic" %
               (args.videoFilePath, args.nDataLoaderThread, args.audioFilePath))
    subprocess.call(command, shell=True, stdout=None)

    # Extract frames
    command = ("ffmpeg -y -i %s -qscale:v 2 -threads %d -f image2 %s -loglevel panic" %
               (args.videoFilePath, args.nDataLoaderThread, os.path.join(args.pyframesPath, '%06d.jpg')))
    subprocess.call(command, shell=True, stdout=None)

    print(f"[2/6] Scene detection...")
    scene = scene_detect(args)

    print(f"[3/6] Face detection...")
    faces = face_detect(args)

    print(f"[4/6] Face tracking...")
    allTracks = []
    for shot in scene:
        if shot[1].frame_num - shot[0].frame_num >= args.minTrack:
            allTracks.extend(track_shot(args, faces[shot[0].frame_num:shot[1].frame_num]))
    print(f"       {len(allTracks)} tracks detected")

    print(f"[5/6] Cropping face clips...")
    vidTracks = []
    for ii, track in enumerate(allTracks):
        vidTracks.append(crop_video(args, track, os.path.join(args.pycropPath, '%05d' % ii)))
    # Save tracks
    savePath = os.path.join(args.pyworkPath, 'tracks.pckl')
    with open(savePath, 'wb') as fil:
        pickle.dump(vidTracks, fil)

    print(f"[6/6] Active Speaker Detection...")
    files = sorted(glob.glob("%s/*.avi" % args.pycropPath))
    scores = evaluate_network(files, args)
    # Save scores
    savePath = os.path.join(args.pyworkPath, 'scores.pckl')
    with open(savePath, 'wb') as fil:
        pickle.dump(scores, fil)

    # Visualization
    output = visualization(vidTracks, scores, args, output_path=pargs.output_path)
    print(f"\n[Done] Output video saved to: {output}")

    # Print summary
    print(f"\n{'='*50}")
    print(f"Speaker Detection Summary:")
    print(f"  Tracks found: {len(vidTracks)}")
    for tidx, score in enumerate(scores):
        avg_score = numpy.mean(score)
        speaking_ratio = numpy.sum(score > 0) / len(score) * 100
        print(f"  Track {tidx}: avg_score={avg_score:.2f}, speaking_ratio={speaking_ratio:.1f}%")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
