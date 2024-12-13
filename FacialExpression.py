import cv2
import gradio as gr
import mediapipe as mp
import numpy as np



def apply_media_pipe_facemesh(image):
    # 初始化 MediaPipe Face Mesh。
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
    # 将输入的图像转换为 RGB。
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 处理图像并获取面部网格关键点。
    results = face_mesh.process(image)
    # 绘制面部网格。
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing.draw_landmarks(
                image, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)
    # 将 RGB 图像转换回 BGR 以供 OpenCV 使用。
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

def magnify_face(image, scale=1.0):
    # 初始化 MediaPipe Face Mesh。
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
    # 将输入的图像转换为 RGB。
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 处理图像并获取面部网格关键点。
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        # 获取人脸关键点的坐标，并转换为图像坐标。
        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')
        for landmark in face_landmarks.landmark:
            x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
            min_x, min_y = min(min_x, x), min(min_y, y)
            max_x, max_y = max(max_x, x), max(max_y, y)

        # 计算边界框的宽度和高度。
        width, height = max_x - min_x, max_y - min_y

        # 计算边界框的中心点。
        center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2

        # 扩大边界框以包含更多头部区域。
        padding = max(width, height) * 0.2  # 增加20%的边距
        padded_min_x = max(0, min_x - padding)
        padded_max_x = min(image.shape[1], max_x + padding)
        padded_min_y = max(0, min_y - padding)
        padded_max_y = min(image.shape[0], max_y + padding)

        # 确保索引是整数
        padded_min_x, padded_max_x, padded_min_y, padded_max_y = int(padded_min_x), int(padded_max_x), int(
            padded_min_y), int(padded_max_y)

        # 裁剪原始图像中扩大的人头区域。
        head_region = image[padded_min_y:padded_max_y, padded_min_x:padded_max_x]

        # 调整人头区域的大小以匹配放大的尺寸。
        resized_head_region = cv2.resize(head_region, (int(width * scale), int(height * scale)))

        # 创建一个新的画布，用于放置放大的人头。
        magnified_canvas = np.zeros((int(height * scale), int(width * scale), 3), dtype=np.uint8)
        # 将调整大小的人头区域复制到新画布上。
        magnified_canvas[0:int(height * scale), 0:int(width * scale)] = resized_head_region

        return magnified_canvas
    return image

class FaceProcessing(object):
    def __init__(self, ui_obj):
        self.name = "Face Image Processing"
        self.description = "Call for Face Image Processing"
        self.ui_obj = ui_obj

    def mp_apply_face_mesh(self, image):
        mesh_image = apply_media_pipe_facemesh(image)
        return mesh_image

    def mp_magnify_photo(self, image):
        magnified_image = magnify_face(image)
        return magnified_image

    def mp_analyze_face_presence(self, image):
        pass

    def mp_analyze_face_intensity(self, image):
        pass

    def mp_analyze_face_expressions(self, image):
        pass
        # return expressions

    def create_ui(self):
        with self.ui_obj:
            gr.Markdown("Facial Expression Analysis with Libreface")
            with gr.Tabs():
                with gr.TabItem("Pipeline Info"):
                    with gr.Row():
                        mp_image_in = gr.Image(label="Image Visualizer" )
                        mp_fm_photo_out = gr.Image(label="MediaPipe Landmark Visualizer" )
                        mp_photo_out = gr.Image(label="Image Visualizer")
                    with gr.Row():
                        mp_photo_action_none = gr.Button("None")
                        mp_apply_fm_action = gr.Button("Apply Face Mesh the Photo")
                        mp_magnify_photo_action = gr.Button("Magnify the Photo")
                    with gr.Row():
                        mp_presence_photo_out = gr.Image(label="Action Unit Presence Visualizer")
                        mp_intensity_photo_out = gr.Image(label="Action Unit Intensity Visualizer")
                        mp_expression_photo_out = gr.Image(label="Facial Expression Visualizer")
                    with gr.Row():
                        mp_analyze_presence_action = gr.Button("Analyze Action Unit Presence")
                        mp_analyze_intensity_action = gr.Button("Analyze Action Unit Intensity")
                        mp_analyze_expression_action = gr.Button("Analyze Facial Expression")

            mp_apply_fm_action.click(
                self.mp_apply_face_mesh,
                [
                    mp_image_in
                ],
                [
                    mp_fm_photo_out
                ]
            )

            mp_magnify_photo_action.click(
                self.mp_magnify_photo,
                [
                    mp_image_in
                ],
                [
                    mp_photo_out
                ]
            )

            mp_analyze_presence_action.click(
                self.mp_analyze_face_presence,
                [
                    mp_image_in
                ],
                [
                    mp_presence_photo_out
                ]
            )

            mp_analyze_intensity_action.click(
                self.mp_analyze_face_intensity,
                [
                    mp_image_in
                ],
                [
                    mp_intensity_photo_out
                ]
            )

            mp_analyze_expression_action.click(
                self.mp_analyze_face_expressions,
                [
                    mp_image_in
                ],
                [
                    mp_expression_photo_out
                ]
            )


    def launch_ui(self):
        self.ui_obj.launch()

if __name__ == '__main__':
    my_app = gr.Blocks()
    face_ui = FaceProcessing(my_app)
    face_ui.create_ui()
    face_ui.launch_ui()