import serial
import serial.tools.list_ports

# List all available ports
ports = list(serial.tools.list_ports.comports())
for port in ports:
    print(f"Available Port: {port}")