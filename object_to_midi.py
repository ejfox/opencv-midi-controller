import cv2
import mido
from mido import Message
import numpy as np

# 🎛️ Tweakable Parameters
CONFIDENCE_THRESHOLD = 0.85
VIDEO_RESOLUTION = (1920, 1080)  # Assuming a resolution of 1920x1080 for scaling
MIDI_PORT_NAME = 'Python Midi Output'
frame_skip = 2  # Adjust this value to skip more or fewer frames

# 🎥 Load YOLO for Object Detection
# net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
layer_names = net.getLayerNames()
output_layers = net.getUnconnectedOutLayersNames()  # 👈 Updated line here

def send_midi_cc(outport, x, y, channel):
    """🎵 Send MIDI CC messages based on object positions."""
    # 📏 Scale x and y to MIDI CC range (0-127)
    cc_x = int(x / VIDEO_RESOLUTION[0] * 127)
    cc_y = int(y / VIDEO_RESOLUTION[1] * 127)
    print(f"cc_x: {cc_x}, cc_y: {cc_y}")  # Debug print statement
    # 🔒 Clamp the values to ensure they are within the valid MIDI range
    cc_x = max(0, min(cc_x, 127))
    cc_y = max(0, min(cc_y, 127))
    control_x = max(0, min(channel, 127))  # Clamp control number for X
    control_y = max(0, min(channel + 1, 127))  # Clamp control number for Y
    # 🎶 Send MIDI CC messages
    outport.send(Message('control_change', control=control_x, value=cc_x, channel=0))  # CC for X on MIDI channel 1
    outport.send(Message('control_change', control=control_y, value=cc_y, channel=0))  # CC for Y on MIDI channel 1
    # Return the CC values
    return cc_x, cc_y

def process_frame(outport, frame, current_channel):
  """🖼️ Process each frame for object detection and send MIDI CC."""
  height, width, channels = frame.shape
  # 🧠 Prepare the frame for object detection
  blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
  net.setInput(blob)
  outs = net.forward(output_layers)
  # 🕵️‍♀️ Detect objects in the frame
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
        # 🎨 Draw the bounding box on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cc_x, cc_y = send_midi_cc(outport, center_x, center_y, current_channel)
        cv2.putText(frame, f"CC: ({cc_x}, {cc_y}) Ch: {current_channel}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        # Increment the current MIDI channel for the next object
        current_channel = (current_channel + 2) % 128  # Keep current_channel within 0 to 127
  # 🖼️ Show the frame with bounding boxes
  cv2.imshow('Video with Bounding Boxes', frame)
  # ⏭️ Press 'q' to close the window
  if cv2.waitKey(1) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
    exit()
  # Return the updated current MIDI channel
  return current_channel

def main():
  """🎬 Main function to process the video and send MIDI CC."""
  with mido.open_output(MIDI_PORT_NAME, virtual=True) as outport:
    cap = cv2.VideoCapture('short_clip.mp4')
    current_channel = 1  # Set the initial MIDI channel to 1
    frame_count = 0  # Initialize frame_count to 0
    while(cap.isOpened()):
      ret, frame = cap.read()
      if ret:
        # 📐 Resize the frame to a lower resolution
        resized_frame = cv2.resize(frame, (640, 360))  # New resolution: 640x360
        # 🔄 Process each resized frame
        if frame_count % frame_skip == 0:
          current_channel = process_frame(outport, resized_frame, current_channel)
        frame_count += 1
      else:
        # 🛑 Stop when the video ends
        break
    # 🧹 Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
  main()

