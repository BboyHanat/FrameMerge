from core.frame_merge import FrameMerge
from config import conf


video_output_list = ['', '']
fm = FrameMerge(**conf['perspctive_conf'])
fm.get_homography_and_mask_all()
print("OKOK")
