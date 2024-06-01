import time
from pymavlink import mavutil
import cv2
import os
from ultralytics import YOLO
from ultralytics.solutions import object_counter
from datetime import datetime

# Initialize MAVLink connection
def init_mavlink_connection():
    master = mavutil.mavlink_connection('udp:127.0.0.1:14550')
    master.wait_heartbeat()
    return master

# Takeoff function
def takeoff(master, altitude):
    master.arducopter_arm()
    master.mav.command_long_send(
        master.target_system, 
        master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0, 0, 0, 0, 0, 0, 0, altitude
    )
    print(f"Taking off to altitude: {altitude} meters")
    time.sleep(10)

# Function to upload predefined path to Pixhawk
def upload_path(master, waypoints):
    for i, waypoint in enumerate(waypoints):
        lat, lon, alt = waypoint
        master.mav.mission_item_send(
            master.target_system,
            master.target_component,
            i, 3, mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 0, 1, 0, 0, 0, 0,
            lat, lon, alt
        )
    master.mav.mission_count_send(master.target_system, master.target_component, len(waypoints))
    print("Path uploaded to Pixhawk")

# Function to change drone altitude
def change_altitude(master, altitude):
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_CONTINUE_AND_CHANGE_ALT,
        0, 0, 0, 0, 0, 0, 0, altitude
    )
    print(f"Changing altitude to {altitude} meters")
    time.sleep(5)

# Function to change drone position
def change_position(master, lat, lon, alt):
    master.mav.set_position_target_global_int_send(
        0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
        0b110111111000, int(lat * 1e7), int(lon * 1e7), alt,
        0, 0, 0, 0, 0, 0, 0, 0
    )
    print(f"Changing position to lat: {lat}, lon: {lon}, alt: {alt} meters")
    time.sleep(5)

# Function for object counting and identification
def process_image(frame, frame_count, fps, output_dir, log_file, object_counts, unique_objects, counter, model):
    timestamp = frame_count / fps
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results = model.track(frame, persist=True, show=False)

    target_detected = False
    target_coords = None

    for r in results:
        for box in r.boxes:
            class_id = int(box.cls)
            object_name = model.names[class_id]
            object_id = int(box.id)

            if object_id not in unique_objects:
                unique_objects.add(object_id)
                if object_name not in object_counts:
                    object_counts[object_name] = 0
                object_counts[object_name] += 1
                margin = 20
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1, x2, y2 = max(0, x1 - margin), max(0, y1 - margin), min(frame.shape[1], x2 + margin), min(frame.shape[0], y2 + margin)
                roi = frame[y1:y2, x1:x2]
                new_object_frame_path = os.path.join(output_dir, f"new_object_{object_name}_{frame_count}.jpg")
                cv2.imwrite(new_object_frame_path, roi)
                print(f"New unique object '{object_name}' detected and saved at frame {frame_count}, time {timestamp:.2f}s, date {current_time}.")
                log_file.write(f"{frame_count}, {current_time}, {object_name}\n")

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf.item()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"{object_name} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            if object_name == 'target_object':  # Replace 'target_object' with the actual name of your target
                target_detected = True
                target_coords = (x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2)

    frame = counter.start_counting(frame, results)
    return frame, target_detected, target_coords

# Function to capture image using RPI camera
def capture_image():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    return frame

# Function to deploy payload using servo motor
def deploy_payload():
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(18, GPIO.OUT)
    pwm = GPIO.PWM(18, 50)
    pwm.start(7.5)
    pwm.ChangeDutyCycle(12.5)
    time.sleep(1)
    pwm.ChangeDutyCycle(7.5)
    pwm.stop()
    GPIO.cleanup()
    print("Payload deployed")

# Function to check if the coordinates are within the geofence
def is_within_geofence(lat, lon, geofence_bounds):
    lat_min, lon_min, lat_max, lon_max = geofence_bounds
    return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max

# Main control loop
def main():
    master = init_mavlink_connection()
    takeoff(master, 15)
    
    predefined_path = [(37.7749, -122.4194, 15), (37.7750, -122.4195, 15)]
    upload_path(master, predefined_path)

    geofence_bounds = (37.7740, -122.4200, 37.7760, -122.4180)  # Define the geofenced area (lat_min, lon_min, lat_max, lon_max)

    model = YOLO("yolov8n.pt")
    model.conf = 0.9

    output_dir = "optxt/refrence7"
    os.makedirs(output_dir, exist_ok=True)

    counter = object_counter.ObjectCounter()
    counter.set_args(view_img=False, reg_pts=[(0, 0), (640, 0), (640, 480), (0, 480)], classes_names=model.names, draw_tracks=True, line_thickness=2)

    object_counts = {}
    unique_objects = set()
    frame_count = 0

    log_path = os.path.join(output_dir, "detection_times7.txt")
    with open(log_path, "w") as log_file:
        log_file.write("Object Detection Log:\n")
        log_file.write("Frame Number, Timestamp, Object Name\n")

        while True:
            frame = capture_image()
            if frame is None:
                print("Failed to capture frame.")
                continue

            frame_count += 1
            frame, target_detected, target_coords = process_image(frame, frame_count, 30, output_dir, log_file, object_counts, unique_objects, counter, model)

            if target_detected:
                current_lat, current_lon = 37.7749, -122.4194  # Use GPS data here to get the current drone position
                target_lat, target_lon = current_lat, current_lon  # Replace with actual calculation based on target_coords

                if is_within_geofence(target_lat, target_lon, geofence_bounds):
                    change_position(master, target_lat, target_lon, 15)
                    change_altitude(master, 5)
                    deploy_payload()
                    break
                else:
                    print("Target detected outside geofence bounds.")
                    break

            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) == ord('q'):
                break

    cv2.destroyAllWindows()

    output_counts_path = os.path.join(output_dir, "refrence07.txt")
    with open(output_counts_path, "w") as file:
        for object_name, count in object_counts.items():
            file.write(f"{object_name}: {count}\n")

    print(f"Object counts have been saved to {output_counts_path}")
    print(f"Detection times have been saved to {log_path}")
    print(f"Total frames processed: {frame_count}")

if __name__ == "__main__":
    main()

