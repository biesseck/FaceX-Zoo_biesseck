"""
@author: Yinglu Liu, Jun Wang
@date: 20201012
@contact: jun21wangustc@gmail.com
"""

from face_masker import FaceMasker

if __name__ == '__main__':
    is_aug = False

    # original
    image_path = 'Data/test-data/test1.jpg'
    face_lms_file = 'Data/test-data/test1_landmark.txt'
    template_name = '7.png'
    masked_face_path = 'test1_mask1.jpg'

    # # Bernardo
    # image_path = '/home/bjgbiesseck/GitHub/FaceX-Zoo_biesseck/face_sdk/api_usage/test_images/FLORENCE_frame_0208.jpg'
    # face_lms_file = '/home/bjgbiesseck/GitHub/FaceX-Zoo_biesseck/face_sdk/api_usage/test_images/FLORENCE_frame_0208_landmark_res0.txt'
    # template_name = '2.png'
    # # template_name = '7.png'
    # masked_face_path = '/home/bjgbiesseck/GitHub/FaceX-Zoo_biesseck/face_sdk/api_usage/test_images/FLORENCE_frame_0208_mask1.jpg'

    # # Bernardo
    # image_path = '/home/bjgbiesseck/GitHub/FaceX-Zoo_biesseck/face_sdk/api_usage/test_images/FLORENCE_frame_0890.jpg'
    # face_lms_file = '/home/bjgbiesseck/GitHub/FaceX-Zoo_biesseck/face_sdk/api_usage/test_images/FLORENCE_frame_0890_landmark_res0.txt'
    # template_name = '7.png'
    # # template_name = '7.png'
    # masked_face_path = '/home/bjgbiesseck/GitHub/FaceX-Zoo_biesseck/face_sdk/api_usage/test_images/FLORENCE_frame_0890_mask1.jpg'

    # # Bernardo
    # image_path = '/home/bjgbiesseck/GitHub/FaceX-Zoo_biesseck/face_sdk/api_usage/test_images/LYHM_00008_2C.png'
    # face_lms_file = '/home/bjgbiesseck/GitHub/FaceX-Zoo_biesseck/face_sdk/api_usage/test_images/LYHM_00008_2C_landmark_res0.txt'
    # template_name = '7.png'
    # # template_name = '7.png'
    # masked_face_path = '/home/bjgbiesseck/GitHub/FaceX-Zoo_biesseck/face_sdk/api_usage/test_images/LYHM_00008_2C_mask1.jpg'

    # Bernardo
    image_path = '/home/bjgbiesseck/GitHub/FaceX-Zoo_biesseck/face_sdk/api_usage/test_images/LYHM_00004_2C.png'
    face_lms_file = '/home/bjgbiesseck/GitHub/FaceX-Zoo_biesseck/face_sdk/api_usage/test_images/LYHM_00004_2C_landmark_res0.txt'
    template_name = '7.png'
    # template_name = '7.png'
    masked_face_path = '/home/bjgbiesseck/GitHub/FaceX-Zoo_biesseck/face_sdk/api_usage/test_images/LYHM_00004_2C_mask1.jpg'

    face_lms_str = open(face_lms_file).readline().strip().split(' ')
    face_lms = [float(num) for num in face_lms_str]
    face_masker = FaceMasker(is_aug)
    face_masker.add_mask_one(image_path, face_lms, template_name, masked_face_path)
