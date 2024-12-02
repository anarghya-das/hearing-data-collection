import serial   #  IMPORTANT: pip install pyserial
import time
import csv
from datetime import datetime

def read_signal(ser):
    try:
        signal = ser.readline().decode('utf-8').strip()
        return signal
    except Exception as e:
        print(f"Error reading signal: {e}")
        return None

def main():
    #IMPORTANT:Port would be different in different system
    # port = "/dev/cu.usbmodem1101"
    port = "COM5"
    # IMPORTANT to check port
    # import serial.tools.list_ports
    #
    # ports = serial.tools.list_ports.comports()
    # for port in ports:
    #     print(f"Device: {port.device}, Description: {port.description}")

    baud_rate = 9600 #this is fixed

    ser = None
    try:
        print(f"Attempting to connect to {port} at {baud_rate} baud rate.")
        ser = serial.Serial(port, baud_rate, timeout=1)
        time.sleep(2)  # Wait for the connection to be established
        print(f"Successfully connected to {port}")

        with open('pulse_data.csv', 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Datetime', 'Timestamp', 'Signal'])

            print("Logging data. Press Ctrl+C to stop.")
            while True:
                signal = read_signal(ser)
                if signal:
                    timestamp = time.time()
                    datetime_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')
                    print(f"Datetime: {datetime_str}, Timestamp: {timestamp}, Signal: {signal}")
                    csvwriter.writerow([datetime_str, timestamp, signal])
                else:
                    print("No signal received.")
    except serial.SerialException as e:
        print(f"Serial error: {e}")
    except KeyboardInterrupt:
        print("Data logging stopped by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if ser and ser.is_open:
            ser.close()
            print(f"Closed connection to {port}")

if __name__ == "__main__":
    main()


