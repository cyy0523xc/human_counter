import sys

if len(sys.argv) < 3:
    print("Usage: $ python {0} [video_path] [output_path] [forbid_box(可选)]",
          sys.argv[0])
    exit()


if __name__ == '__main__':
    from yolo import YOLO
    from yolo import detect_video
    video_path = sys.argv[1]
    output_path = sys.argv[2]
    if len(sys.argv) > 3:
        box = [int(i) for i in sys.argv[3].split(',')]
        box = [box[0:2], box[2:4]]
        detect_video(YOLO(), video_path, output_path, forbid_box=box)
    else:
        detect_video(YOLO(), video_path, output_path)
