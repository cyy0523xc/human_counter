import fire
from people_flow import YOLO
from people_flow import detect_video, detect_img


class Detect:
    def image(self, image_path, score_threshold=0.3):
        model = YOLO(score_threshold=score_threshold)
        detect_img(model, image_path)

    def video(self, video_path, output_path, start=0, end=0,
               score_threshold=0.3):
        """视频识别接口
        Args:
            forbid_box: 禁区设置，格式：x1,y1;x2,y2;x3,y3;x4,y4
        Example:
            python3 client.py video --video-path ../f0662804860a3dfa5774b4b0067a950c.mp4 --output-path out.avi --start 10 --end 25 --score_threshold=0.3
        """
        start, end = int(start), int(end)
        score_threshold = float(score_threshold)
        model = YOLO(score_threshold=score_threshold)
        detect_video(model, video_path, output_path, start=start, end=end)

    def forbidden(self, video_path, output_path, start=0, end=0,
                  score_threshold=0.3, forbid_box=None):
        """视频识别接口：禁区识别
        Args:
            forbid_box: 禁区设置，格式：x1,y1;x2,y2;x3,y3;x4,y4
        Example:
            python3 client.py forbidden --video-path ../f0662804860a3dfa5774b4b0067a950c.mp4 --output-path out.avi --start 15 --end 16 --score_threshold=0.05 --forbid_box="170,640;335,705"
        """
        start, end = int(start), int(end)
        score_threshold = float(score_threshold)
        if forbid_box is not None:
            forbid_box = [b.split(',') for b in forbid_box.split(';')]
            forbid_box = [[int(x), int(y)] for x, y in forbid_box]
            if len(forbid_box) == 2:     # 左上角和右下角
                [x1, y1], [x2, y2] = forbid_box
                forbid_box = [forbid_box[0], [x2, y1], forbid_box[1], [x1, y2]]

        print(forbid_box)
        model = YOLO(score_threshold=score_threshold)
        detect_video(model, video_path, output_path, start=start, end=end,
                     forbid_box=[forbid_box])

    def forbidden_multi(self, video_path, output_path, start=0, end=0,
                        score_threshold=0.3, forbid_box=None):
        """视频识别接口：多禁区识别
        Args:
            forbid_box: 禁区设置，格式：x1,y1;x2,y2;x3,y3;x4,y4
        Example:
            python3 client.py forbidden_multi --video-path ../越界检测.mp4 \
                --output-path out.avi --start 15 --end 16 --score_threshold=0.05 \
                --forbid_box="0,558;93,543;222,480;290,375;255,360;0,487|540,412;540,520;430,479;355,382;420,372"
        """
        start, end = int(start), int(end)
        score_threshold = float(score_threshold)
        if forbid_box is not None:
            forbid_box = [[[int(x) for x in b.split(',')]
                           for b in s.split(';')]
                          for s in forbid_box.split('|')]

        print(forbid_box)
        model = YOLO(score_threshold=score_threshold)
        detect_video(model, video_path, output_path, start=start, end=end,
                     forbid_box=forbid_box)


if __name__ == '__main__':
    fire.Fire(Detect)
