from core.frame_relative import FrameRelative
from config import conf

fm = FrameRelative(**conf['homo_conf'])
fm.read_from_url_and_save()
fm.get_homography_and_mask_all()
print("OKOK")
