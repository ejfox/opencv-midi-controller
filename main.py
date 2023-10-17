import cv2

def main():
    # Load an image from file
    image = cv2.imread('example.jpg')

    # Check if the image was loaded successfully
    if image is None:
        print('Error: Unable to open image file.')
        return

    # Display the image in a window
    cv2.imshow('Hello World', image)

    # Wait for a key press and then close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
