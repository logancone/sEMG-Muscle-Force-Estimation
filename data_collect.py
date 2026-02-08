# pyright: reportPossiblyUnboundVariable=false
import serial

import struct
import argparse

import numpy as np

import time

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui

from collections import deque

from multiprocessing import Process, Queue

parser = argparse.ArgumentParser(description="Data Collection Parameters")

parser.add_argument('--subj_id', type=int, required=True)
parser.add_argument('--mvc_trial', type=int, default=0) #Set to non-zero if MVC trials, set to number of trial
parser.add_argument('--mvc_val', type=int) #The MVC gathered from trials; must set to number if mvc_trial == 0
args = parser.parse_args()

save_folder_norm = "raw_data/raw_data_chunks"
save_folder_mvc = "raw_data/raw_mvc_trials"

running = True

def data_read_loop(q):
    global running

    semgVal = 0
    forceVal = 0


    semgVals = []
    forceIdxs = []
    forceVals = []

    chunk_size = 10000
    chunk_id = 0

    # Set the current active read mode (0 = ramp, 1 = hold, 2 = rest)
    mode = 0

    # Set target mvc (0=35%, 1=50%, 2=65%)
    peak_level = 0
    peak_vals = [35, 50, 65]


    loop_id = 0


    # Open serial port
    ser = serial.Serial('COM3', baudrate=115200, timeout=10)



    # Intro prints
    print(ser.read(2)) #Read \r\n
    print(ser.read(17)) # Read 'Initializing...\r\n'
    print(ser.read(13)) #Read 'Starting...\r\n'


    try:
        print("Beginning!...")

        # Read loop
        while running:
            # Check port is still open
            if ser.is_open == False:
                raise Exception("Serial Port Disconnected. ")
            
            # Read a header
            header = ser.read(1)

            # Semg Header
            if header == b'\xe1':
                semgVal, = struct.unpack('<H', ser.read(2))

                semgVals.append(semgVal)
                

            # Force Header
            elif header == b'\xf1':
                forceIdx, forceVal = struct.unpack('<Ih', ser.read(6))

                forceIdxs.append(forceIdx)
                forceVals.append(forceVal)

                # Print semg and force val
                if args.mvc_trial == 0:
                    
                    normForce = forceVal / args.mvc_val
                    q.put(normForce)
                    
                    loop_id += 1
                    if loop_id == 10: #Slow down prints to make more readable
                        print(f"Semg: {semgVal} | Norm Force: {normForce} | Raw Force: {forceVal}")
                        loop_id = 0            
                                    
                else:
                    # If MVC trial, print semg and raw force
                    print(f"Semg: {semgVal} | Raw Force: {forceVal}")
                

            # Neither: Desync Detection
            else:
                print("BYTE WAS DROPPED!!! MISALIGNMENT!")
                print(header)
                raise Exception("Read non-header as header")
            

            # Check if ~10 sec has passed (10k samples of semg at 1kHz) and only save on normal trial (MVC trial is only ~5 sec long)
            if len(semgVals) >= chunk_size and args.mvc_trial == 0:
                # Save lists as ndarrays
                npSemgVals = np.array(semgVals)
                npForceIdxs = np.array(forceIdxs)
                npForceVals = np.array(forceVals)

                # Save chunk and reset list
                filename = save_folder_norm + f"/Subject_{args.subj_id}_Chunk_{chunk_id}"
                
                np.savez(filename, npSemgVals=npSemgVals, npForceIdxs=npForceIdxs, npForceVals=npForceVals)

                semgVals.clear()
                forceIdxs.clear()
                forceVals.clear()

                print(f"Chunk {chunk_id} Saved!")
                chunk_id += 1
            

    # Exceptions
    except KeyboardInterrupt:
        print("Control+C has been pressed.")
    except Exception as e:
        print(f"Unexpected Error: {e}")
    finally:
        # Graceful shutdown
        # Final Save
        npSemgVals = np.array(semgVals)
        npForceIdxs = np.array(forceIdxs)
        npForceVals = np.array(forceVals)

        # Save chunk and reset list
        if args.mvc_trial == 0:
            filename = save_folder_norm + f"/Subject_{args.subj_id}_Chunk_{chunk_id}"
        else:
            filename = save_folder_mvc + f"/Subject_{args.subj_id}_MVC_Trial_{args.mvc_trial}"
            print(f"Highest force value in this trial: {npForceVals.max()}")

        np.savez(filename, npSemgVals=npSemgVals, npForceIdxs=npForceIdxs, npForceVals=npForceVals)

        semgVals.clear()
        forceIdxs.clear()
        forceVals.clear()

        print(f"Final! Chunk {chunk_id} Saved!")
        
        if ser.is_open:
            ser.close()

        running = False
        print("Shutdown Successfull!")

def plot_data_loop(q):
    global running

    targetValsTemp = []
    sps = 12
    curTargetVal = 0
    targetList = [.35, .5, .65]

    # targetGen (assuming 12 SPS)
    for i in range(3):
        for j in range(sps * 2):
            targetValsTemp.append(curTargetVal)
            curTargetVal += targetList[i] / (sps*2)

        for j in range(sps * 4):
            targetValsTemp.append(targetList[i])

        for j in range(sps):
            targetValsTemp.append(0)
        curTargetVal = 0
    
    targetVals = np.array(targetValsTemp)
    targetVals = np.roll(targetVals, 100)

    # Initialize live plot
    app = pg.mkQApp("Graph")
    win = pg.plot(title="Live Force from Load Plate")

    win.setXRange(0, 200)
    win.setYRange(0, 1)

    forceCurve = win.plot(pen=pg.mkPen(color='w', width=5))
    targetCurve = win.plot(pen=pg.mkPen(color='r', width=2))

    plot_vals = deque([0] * 100, maxlen=100)

    while running == True:
        if not q.empty():
            plot_vals.append(q.get())
            forceCurve.setData(plot_vals)
            targetCurve.setData(targetVals[:200])

            temp = targetVals[0]
            targetVals[:-1] = targetVals[1:]
            targetVals[-1] = temp

            QtGui.QGuiApplication.processEvents() # type: ignore


if __name__ == "__main__":
    q = Queue()
    p1 = Process(target=data_read_loop, args=(q,))
    p2 = Process(target=plot_data_loop, args=(q, ))
    p1.start()
    p2.start()
    p1.join()
    p2.join()