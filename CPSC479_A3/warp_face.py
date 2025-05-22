import cv2
from detect_landmarks import FaceLandmarkDetector
import extrapolate_vector_field as evf
from warp_image import ImageWarper
import numpy as np


def generate_face_warp_video(face1_filename, face2_filename, out_filename, n_frames=30, fps=30):
    # Read face photos. We assume they have the same size
    face1 = cv2.imread(face1_filename)
    face2 = cv2.imread(face2_filename)
    assert face1.shape == face2.shape
    img_size = face1.shape[0:2]

    # Here you will need to tie everything together: landmarks, warp fields and warping.
    # Create sparse field using face landmarks
    mark_detector = FaceLandmarkDetector()
    landmarks_1 = mark_detector.predict(face1)
    landmarks_2 = mark_detector.predict(face2)

    #print(landmarks_1)

    # Subtract landmark 2 from 1 to create sparse warp field
    sparse_warp_delta = landmarks_2 - landmarks_1

    # Interpolate warp field from landmarks_1 and delta
    e = evf.Extrapolator()
    dense_field_x, dense_field_y = e.extrapolate(landmarks_1[:, 0], landmarks_1[:, 1], sparse_warp_delta[:, 0], sparse_warp_delta[:, 1], out_size=img_size)
    out = cv2.VideoWriter(out_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (img_size[1], img_size[0]))

    w = ImageWarper()

    warp_amounts = np.linspace(0., 1., n_frames)
    for i, warp_amount in enumerate(warp_amounts):
        # We alpha blend the original images. Replace this to produce a warping effect
        face_out = w.warp(face2, dense_field_x, dense_field_y, warp_amount=warp_amount) * (1 - warp_amount) + warp_amount * w.warp(face1, -dense_field_x, -dense_field_y, warp_amount=(1 - warp_amount))

        # write video frame
        out.write(face_out.astype(np.uint8))

    out.release()


if __name__ == '__main__':
    face1_filename = './data/head1.jpg'
    face2_filename = './data/head2.jpg'
    out_filename = './data/out.mp4'

    generate_face_warp_video(face1_filename, face2_filename, out_filename)
