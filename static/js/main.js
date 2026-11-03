document.addEventListener('DOMContentLoaded', () => {
    // UI state elements
    const totalVehiclesEl = document.getElementById('total-vehicles');
    const carCountEl = document.getElementById('car-count');
    const bikeCountEl = document.getElementById('bike-count');
    const busCountEl = document.getElementById('bus-count');
    const truckCountEl = document.getElementById('truck-count');
    const densityValEl = document.getElementById('density-value');
    const densityProgressEl = document.getElementById('density-progress');
    const congestionValEl = document.getElementById('congestion-value');
    const congestionConfEl = document.getElementById('congestion-confidence');
    const signalModeEl = document.getElementById('signal-mode');
    const averageWaitEl = document.getElementById('average-wait');
    const signalTimerEl = document.getElementById('signal-timer');
    const fpsValEl = document.getElementById('fps-value');
    
    // Traffic Light elements
    const lightRed = document.getElementById('light-red');
    const lightYellow = document.getElementById('light-yellow');
    const lightGreen = document.getElementById('light-green');
    
    // Toggle Buttons
    const btnRain = document.getElementById('btn-rain');
    const btnNight = document.getElementById('btn-night');
    const btnEmergency = document.getElementById('btn-emergency');
    const btnReset = document.getElementById('btn-reset');
    
    // Alert Banners
    const bannerEmergency = document.getElementById('banner-emergency');
    const bannerAccident = document.getElementById('banner-accident');

    // System config tracking
    let systemState = {
        rainMode: false,
        nightMode: false,
        emergencyOverride: false
    };

    // Update Dashboard DOM metrics
    function updateDashboard(data) {
        if (!data) return;

        // Sync values
        if (totalVehiclesEl) totalVehiclesEl.textContent = data.vehicle_count;
        if (carCountEl) carCountEl.textContent = data.car_count;
        if (bikeCountEl) bikeCountEl.textContent = data.bike_count;
        if (busCountEl) busCountEl.textContent = data.bus_count;
        if (truckCountEl) truckCountEl.textContent = data.truck_count;
        if (densityValEl) densityValEl.textContent = data.density + '%';
        if (averageWaitEl) averageWaitEl.textContent = data.average_wait + 's';
        if (signalTimerEl) signalTimerEl.textContent = data.signal_timer + 's';
        if (signalModeEl) signalModeEl.textContent = data.signal_mode;
        if (fpsValEl) fpsValEl.textContent = 'FPS: ' + data.fps;

        // Density progress bar width & color class
        if (densityProgressEl) {
            densityProgressEl.style.width = data.density + '%';
            densityProgressEl.className = 'progress-bar'; // reset
            if (data.density > 75) {
                densityProgressEl.classList.add('bg-danger');
            } else if (data.density > 45) {
                densityProgressEl.classList.add('bg-warning');
            } else {
                densityProgressEl.classList.add('bg-success');
            }
        }

        // Congestion label and badge classes
        if (congestionValEl) {
            congestionValEl.textContent = data.congestion_level;
            
            // Highlight text color
            congestionValEl.className = 'fw-bold'; // reset
            if (data.congestion_level === 'Heavy Traffic') {
                congestionValEl.classList.add('text-danger');
            } else if (data.congestion_level === 'Moderate Traffic') {
                congestionValEl.classList.add('text-warning');
            } else if (data.congestion_level === 'Light Traffic') {
                congestionValEl.classList.add('text-info');
            } else {
                congestionValEl.classList.add('text-success');
            }
        }
        
        if (congestionConfEl) {
            congestionConfEl.textContent = 'Confidence: ' + data.congestion_confidence + '%';
        }

        // Traffic Light Signal visual sync
        if (lightRed && lightYellow && lightGreen) {
            lightRed.classList.remove('active');
            lightYellow.classList.remove('active');
            lightGreen.classList.remove('active');

            if (data.active_signal === 'RED') {
                lightRed.classList.add('active');
            } else if (data.active_signal === 'YELLOW') {
                lightYellow.classList.add('active');
            } else if (data.active_signal === 'GREEN') {
                lightGreen.classList.add('active');
            }
        }

        // Sync buttons toggles visually from state
        systemState.rainMode = data.rain_mode;
        systemState.nightMode = data.night_mode;
        systemState.emergencyOverride = data.emergency_override;

        toggleButtonActive(btnRain, data.rain_mode);
        toggleButtonActive(btnNight, data.night_mode);
        toggleButtonActive(btnEmergency, data.emergency_override);

        // Emergency and Accident alert banner triggers
        if (bannerEmergency) {
            if (data.emergency_detected || data.emergency_override) {
                bannerEmergency.classList.remove('d-none');
            } else {
                bannerEmergency.classList.add('d-none');
            }
        }

        if (bannerAccident) {
            if (data.accident_detected) {
                bannerAccident.classList.remove('d-none');
            } else {
                bannerAccident.classList.add('d-none');
            }
        }
    }

    function toggleButtonActive(buttonEl, isActive) {
        if (!buttonEl) return;
        if (isActive) {
            buttonEl.classList.add('active');
        } else {
            buttonEl.classList.remove('active');
        }
    }

    // Server update trigger helper
    function updateServerState(payload) {
        fetch('/process', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'updated') {
                updateDashboard(data.state);
            }
        })
        .catch(err => console.error("Error setting server state:", err));
    }

    // Toggle Click Event Listeners
    if (btnRain) {
        btnRain.addEventListener('click', () => {
            const nextVal = !systemState.rainMode;
            updateServerState({ "rain_mode": nextVal });
        });
    }

    if (btnNight) {
        btnNight.addEventListener('click', () => {
            const nextVal = !systemState.nightMode;
            updateServerState({ "night_mode": nextVal });
        });
    }

    if (btnEmergency) {
        btnEmergency.addEventListener('click', () => {
            const nextVal = !systemState.emergencyOverride;
            updateServerState({ "emergency_override": nextVal });
        });
    }

    if (btnReset) {
        btnReset.addEventListener('click', () => {
            if (confirm("Reset vehicle counters? This restarts cumulative data logging.")) {
                fetch('/reset_counts', { method: 'POST' })
                .then(res => res.json())
                .then(data => {
                    if (data.status === 'reset') {
                        console.log("Counters reset.");
                    }
                });
            }
        });
    }

    // Polling thread (runs every 600ms)
    function pollData() {
        fetch('/process')
            .then(res => res.json())
            .then(data => {
                updateDashboard(data);
            })
            .catch(err => console.error("Polling error:", err));
    }

    // Start polling loop
    setInterval(pollData, 600);
    pollData(); // initial call
});
