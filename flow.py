import fire
from people_flow import YOLO
from people_flow import detect_video


class Video:
    def simple(self, video_path, output_path, start=0, end=0,
               score_threshold=0.5):
        """视频识别接口
        Args:
            forbid_box: 禁区设置，格式：x1,y1;x2,y2;x3,y3;x4,y4

        Example:
            python3 flow.py simple --video-path ../f0662804860a3dfa5774b4b0067a950c.mp4 --output-path out.avi --start 10 --end 25 --score_threshold=0.3
        """
        detect_video(YOLO(), video_path, output_path, start=start, end=end,
                     score_threshold=score_threshold)

    def forbidden(self, video_path, output_path, start=0, end=0,
                  score_threshold=0.5, forbid_box=None):
        """视频识别接口：禁区识别
        Args:
            forbid_box: 禁区设置，格式：x1,y1;x2,y2;x3,y3;x4,y4
        """
        if forbid_box is not None:
            forbid_box = [b.split(',') for b in forbid_box.split(';')]

        detect_video(YOLO(), video_path, output_path, start=start, end=end,
                     forbid_box=forbid_box, score_threshold=score_threshold)


if __name__ == '__main__':
    fire.Fire(Video)
