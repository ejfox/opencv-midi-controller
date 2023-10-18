import cv2
import mido
from mido import Message
import numpy as np

# 🎛️ Tweakable Parameters
CONFIDENCE_THRESHOLD = 0.73
VIDEO_RESOLUTION = (1920, 1080)  # Assuming a resolution of 1920x1080 for scaling
MIDI_PORT_NAME = 'Python Midi Output'
frame_skip = 2  # Adjust this value to skip more or fewer frames
MAX_OBJECTS = 2  # New variable to limit the number of detected objects

with open("coco.names", "r") as f:
    labels = f.read().strip().split("\n")

# 🎥 Load YOLO for Object Detection
# net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
layer_names = net.getLayerNames()
output_layers = net.getUnconnectedOutLayersNames()  # 👈 Updated line here


def update_trackers(frame):
    global trackers
    # Run your object detection code here
    # ... (object detection code)
    # Reset the multi-tracker
    trackers = cv2.MultiTracker_create()
    for box in boxes:  # Assume boxes is a list of bounding boxes from object detection
        tracker = cv2.TrackerCSRT_create()  # Or any other OpenCV tracker
        trackers.add(tracker, frame, box)


# 🎵 Send MIDI CC messages based on object positions.
# 📏 Scale x and y to MIDI CC range (0-127)
# 🔒 Clamp the values to ensure they are within the valid MIDI range
# 🎶 Send MIDI CC messages
# Return the CC values
def send_midi_cc(outport, x, y, object_channel):
    """🎵 Send MIDI CC messages based on object positions."""
    # 📏 Scale x and y to MIDI CC range (0-127)
    cc_x = int(x / VIDEO_RESOLUTION[0] * 127)
    cc_y = int(y / VIDEO_RESOLUTION[1] * 127)
    # print(f"cc_x: {cc_x}, cc_y: {cc_y}")  # Debug print statement
    # 🔒 Clamp the values to ensure they are within the valid MIDI range
    cc_x = max(0, min(cc_x, 127))
    cc_y = max(0, min(cc_y, 127))
    control_x = max(0, min(object_channel, 127))  # Clamp control number for X
    control_y = max(0, min(object_channel + 1, 127))  # Clamp control number for Y
    # 🎶 Send MIDI CC messages
    outport.send(Message('control_change', control=control_x, value=cc_x, channel=object_channel))  # CC for X on object_channel
    outport.send(Message('control_change', control=control_y, value=cc_y, channel=object_channel))  # CC for Y on object_channel
    # Return the CC values
    return cc_x, cc_y

def process_frame(outport, frame, current_channel):
    """🖼️ Process each frame for object detection and send MIDI CC."""
    global trackers
    height, width, channels = frame.shape
    # 🧠 Prepare the frame for object detection
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    object_channel = 1  # Initialize object_channel to 1
    object_count = 0  # Initialize object_count to 0

    # 🕵️‍♀️ Detect objects in the frame
    boxes = []
    confidences = []
    class_ids = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # 🎯 Only consider confident detections
            if confidence > CONFIDENCE_THRESHOLD:
                # 🎈 Get the bounding box coordinates and dimensions
                box = detection[0:4] * np.array([width, height, width, height])
                (center_x, center_y, w, h) = box.astype("int")
                x = int(center_x - (w / 2))
                y = int(center_y - (h / 2))
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # 🎯 Apply non-maximum suppression to remove overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, 0.4)

    # 🎨 Draw the bounding box on the frame
    for i in range(len(boxes)):
        if i in indexes:
            label = labels[class_ids[i]]
            # print(label)
            x, y, w, h = boxes[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 4)
            if object_count < MAX_OBJECTS:  # Only send MIDI CC if object_count is less than MAX_OBJECTS
                cc_x, cc_y = send_midi_cc(outport, center_x, center_y, object_channel)  # Use object_channel instead of current_channel
                cv2.putText(frame, f"{label} ({cc_x}, {cc_y}) Ch: {object_channel}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                object_count += 1
            # Increment the object_channel for the next object, and reset to 1 if it exceeds 127
            object_channel += 1
            if object_channel > 127:
                object_channel = 1

    # Return the updated current MIDI channel
    return current_channel
    
def main():
    """🎬 Main function to process the video and send MIDI CC."""
    with mido.open_output(MIDI_PORT_NAME, virtual=True) as outport:
        cap = cv2.VideoCapture(0)
        current_channel = 1  # Set the initial MIDI channel to 1
        frame_count = 0  # Initialize frame_count to 0
        while(True):
            ret, frame = cap.read()
            if ret:
                # 📐 Resize the frame to a lower resolution
                resized_frame = cv2.resize(frame, (640, 360))  # New resolution: 640x360
                # 🔄 Process each resized frame
                # if frame_count % frame_skip == 0:
                current_channel = process_frame(outport, frame, current_channel)  # Pass frame instead of resized_frame
                frame_count += 1
                # 🎥 Display the frame
                cv2.imshow('Frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                # 🛑 Stop when the video ends
                break
        # 🧹 Clean up
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()