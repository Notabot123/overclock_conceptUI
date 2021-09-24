from midiutil import MIDIFile
import random
import pandas as pd
import simpy

import numpy as np
import matplotlib.pyplot as plt

class Song_struct:
    def __init__(self,env):
        self.env = env
        self.note = env.event()
        self.arising = None
        
        self.numnotes = 3
        self.phrases = 5 
        
        self.cumnotes = 0
        self.cumphrases = 0
        
        self.num_procs = [env.process(self.note_play(idx)) for idx in range(self.numnotes)]
        self.days_proc = env.process(self.phrase())       
        
        
    def note_play(self,idx):
        for i in range(self.phrases):
            print(" note num: " + str(i) + ", for phrase: " + str(idx) +". ", end='')
            yield self.note
            self.cumnotes += 1
        return random.randint(1,100)
    def phrase(self):
        for i in range(self.phrases):
            yield self.env.timeout(45)
            self.note.succeed()
            self.note = self.env.event()
            print("...lick: " + str(i))
            self.cumphrases += 1
        return 'hi there'

env = simpy.Environment()
fleet = Song_struct(env)
env.run()


degrees  = [60, 62, 64, 65, 67, 69, 71, 72]  # MIDI note number
track    = 0
channel  = 0
time     = 0    # In beats
duration = 1    # In beats
tempo    = 60   # In BPM
volume   = 100  # 0-127, as per the MIDI standard

MyMIDI = MIDIFile(1)  # One track, defaults to format 1 (tempo track is created
                      # automatically)
MyMIDI.addTempo(track, time, tempo)

for i, pitch in enumerate(degrees):
    MyMIDI.addNote(track, channel, pitch, time + i, duration, volume)

with open("major-scale.mid", "wb") as output_file:
    MyMIDI.writeFile(output_file)