class SignalController:
    def __init__(self):
        self.current_state = "GREEN"  # "GREEN", "YELLOW", "RED"
        self.time_remaining = 30.0    # Seconds remaining in current state
        
        # Base limits
        self.min_green = 20.0
        self.max_green = 90.0
        
    def calculate_timings(self, density_pct, emergency_active=False, rain_mode=False, night_mode=False):
        """
        Dynamically calculates Green, Yellow, and Red durations based on density 
        and environment settings.
        
        Density rules:
        - Density < 20% -> Green: 20s, Yellow: 5s, Red: 20s
        - Density > 60% -> Green: 60s, Yellow: 5s, Red: 15s
        - Density > 90% -> Green: 90s, Yellow: 5s, Red: 10s
        """
        # 1. Emergency vehicle override (highest priority)
        if emergency_active:
            # Force max green for the lane containing the emergency vehicle
            green = 90.0
            yellow = 5.0
            red = 5.0
            return {
                "green_time": green,
                "yellow_time": yellow,
                "red_time": red,
                "mode": "EMERGENCY OVERRIDE"
            }
            
        # 2. Base timings based on density (with linear interpolation)
        if density_pct <= 20.0:
            green = 20.0
            yellow = 5.0
            red = 20.0
        elif density_pct <= 60.0:
            # Interpolate between 20% and 60%
            factor = (density_pct - 20.0) / 40.0
            green = 20.0 + factor * (60.0 - 20.0)
            yellow = 5.0
            red = 20.0 - factor * (20.0 - 15.0)
        elif density_pct <= 90.0:
            # Interpolate between 60% and 90%
            factor = (density_pct - 60.0) / 30.0
            green = 60.0 + factor * (90.0 - 60.0)
            yellow = 5.0
            red = 15.0 - factor * (15.0 - 10.0)
        else:
            green = 90.0
            yellow = 5.0
            red = 10.0
            
        # 3. Apply Weather/Time Modifiers
        mode = "ADAPTIVE"
        
        if rain_mode:
            # Rain mode: Increase yellow time for braking safety, adjust red
            yellow += 2.0
            red += 2.0
            mode += " + RAIN"
            
        if night_mode:
            # Night mode: Moderate green, slightly longer yellow, shorter red
            green = max(20.0, min(green, 30.0))  # Cap green to prevent long empty waits
            yellow += 1.0
            red = max(10.0, red - 3.0)
            mode += " + NIGHT"
            
        return {
            "green_time": round(green, 1),
            "yellow_time": round(yellow, 1),
            "red_time": round(red, 1),
            "mode": mode
        }
        
    def get_signal_durations(self, density_pct, emergency_active=False, rain_mode=False, night_mode=False):
        """
        Convenience wrapper returning calculated timings.
        """
        return self.calculate_timings(density_pct, emergency_active, rain_mode, night_mode)
