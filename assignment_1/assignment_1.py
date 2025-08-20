import cv2

def print_image_information(image):
    img = cv2.imread(image)

    if img is None:
        print("Could not open image.")
        exit(1)

    height, width, channel = img.shape
    size = img.size
    data_type = img.dtype
    print("Height:", height)
    print("Width:", width)
    print("Channel:", channel)
    print("Size:", size, img.shape)
    print("DataType:", data_type)

    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def print_video_information():
    cam = cv2.VideoCapture(0)

    if cam.isOpened() is False:
        print("Could not open camera.")
        exit(1)

    fps = cam.get(cv2.CAP_PROP_FPS)
    width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)

    with open("solutions/camera_outputs.txt", "w") as f:
        f.write(f"fps: {fps}\n")
        f.write(f"width: {width}\n")
        f.write(f"height: {height}\n")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Could not read camera frame.")
            exit(1)

        cv2.imshow('img', frame)
        if cv2.waitKey(1) == ord('x'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print_image_information('lena.png')
    print_video_information()