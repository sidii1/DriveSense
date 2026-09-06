import cv2
import numpy as np
import os

def generate_road_video(output_path, duration_sec=10, fps=30):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    width, height = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    num_frames = duration_sec * fps
    
    # Road geometry
    vanishing_point = (640, 380)
    left_bottom = (150, 720)
    left_top = (580, 390)
    right_bottom = (1130, 720)
    right_top = (700, 390)
    
    # Car simulation parameters
    car_x = 640
    car_y_start = 390
    car_y_end = 650
    
    for frame_idx in range(num_frames):
        # Create canvas (dark sky and green terrain)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Terrain (green grass)
        cv2.rectangle(frame, (0, 380), (width, height), (30, 120, 30), -1)
        # Sky (gradient-like dark blue to black)
        for y in range(0, 380):
            color = int(10 + (y / 380.0) * 40)
            frame[y, :] = [color + 10, color, 5]
            
        # Draw road (dark gray polygon)
        road_pts = np.array([left_bottom, left_top, right_top, right_bottom], np.int32)
        cv2.fillPoly(frame, [road_pts], (50, 50, 50))
        
        # Draw solid yellow left lane line
        cv2.line(frame, left_bottom, left_top, (0, 200, 200), 8, cv2.LINE_AA)
        
        # Draw solid white right lane line
        cv2.line(frame, right_bottom, right_top, (230, 230, 230), 8, cv2.LINE_AA)
        
        # Draw moving center dashed line (white)
        # Dashes slide down with frame index to simulate motion
        dash_speed = 10
        offset = (frame_idx * dash_speed) % 80
        for y in range(390, 720, 80):
            y_curr = y + offset
            if y_curr > 720:
                continue
            
            # Map perspective depth to screen coordinates
            t1 = (y_curr - 390) / (720 - 390)
            t2 = ((y_curr + 40) - 390) / (720 - 390)
            if t2 > 1.0: t2 = 1.0
            
            cx1 = int(vanishing_point[0] + (640 - vanishing_point[0]) * t1)
            cy1 = int(vanishing_point[1] + (720 - vanishing_point[1]) * t1)
            
            cx2 = int(vanishing_point[0] + (640 - vanishing_point[0]) * t2)
            cy2 = int(vanishing_point[1] + (720 - vanishing_point[1]) * t2)
            
            # Line thickness increases as it gets closer
            thickness = max(2, int(6 * t1))
            cv2.line(frame, (cx1, cy1), (cx2, cy2), (255, 255, 255), thickness, cv2.LINE_AA)
            
        # Draw a simulated car moving in the right lane (ahead of us)
        # It oscillates closer and farther
        progress = (np.sin(frame_idx * 0.05) + 1.0) / 2.0  # 0 to 1
        car_y = int(car_y_start + progress * (car_y_end - car_y_start))
        
        # Perspective scaling of the car size
        t_car = (car_y - 390) / (720 - 390)
        car_width = int(40 + t_car * 180)
        car_height = int(25 + t_car * 110)
        
        # Calculate horizontal center in right lane
        car_x = int(600 + t_car * 300)
        
        # Draw vehicle body (gray metallic sedan)
        car_top_left = (car_x - car_width // 2, car_y - car_height)
        car_bottom_right = (car_x + car_width // 2, car_y)
        
        # Draw car shadow
        shadow_center = (car_x, car_y)
        cv2.ellipse(frame, shadow_center, (car_width // 2 + 10, int(10 * t_car + 2)), 0, 0, 360, (20, 20, 20), -1)
        
        # Main body
        cv2.rectangle(frame, car_top_left, car_bottom_right, (100, 100, 110), -1)
        cv2.rectangle(frame, car_top_left, car_bottom_right, (70, 70, 75), max(1, int(3 * t_car)))
        
        # Car roof/cabin (dark glass area)
        roof_y_start = car_y - car_height
        roof_y_end = car_y - int(car_height * 0.5)
        roof_w_offset = int(car_width * 0.15)
        roof_pts = np.array([
            [car_top_left[0] + roof_w_offset, roof_y_start],
            [car_bottom_right[0] - roof_w_offset, roof_y_start],
            [car_bottom_right[0] - int(roof_w_offset * 0.5), roof_y_end],
            [car_top_left[0] + int(roof_w_offset * 0.5), roof_y_end]
        ], np.int32)
        cv2.fillPoly(frame, [roof_pts], (30, 30, 35))
        
        # Tail lights (red rectangles)
        light_w = int(car_width * 0.15)
        light_h = int(car_height * 0.1)
        light_y = car_y - int(car_height * 0.3)
        # Left taillight
        cv2.rectangle(frame, (car_top_left[0] + 5, light_y), (car_top_left[0] + 5 + light_w, light_y + light_h), (0, 0, 255), -1)
        # Right taillight
        cv2.rectangle(frame, (car_bottom_right[0] - 5 - light_w, light_y), (car_bottom_right[0] - 5, light_y + light_h), (0, 0, 255), -1)
        
        # License plate (yellow)
        plate_w = int(car_width * 0.25)
        plate_h = int(car_height * 0.08)
        plate_x = car_x - plate_w // 2
        plate_y = car_y - int(car_height * 0.2)
        cv2.rectangle(frame, (plate_x, plate_y), (plate_x + plate_w, plate_y + plate_h), (100, 220, 255), -1)
        
        # Write frame
        out.write(frame)
        
    out.release()
    print(f"Synthetic video successfully saved to: {output_path}")


if __name__ == "__main__":
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "real_road.mp4.f401.mp4")
    generate_road_video(base)
