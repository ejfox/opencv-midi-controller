import mido
from mido import Message
import time

def send_midi_note():
    port_name = 'Python Midi Output'

    with mido.open_output(port_name, virtual=True) as outport:
        while True:
            note_on = Message('note_on', note=24, velocity=64, channel=0)
            outport.send(note_on)
            time.sleep(1)  # pause for 1 second

if __name__ == "__main__":
    send_midi_note()
