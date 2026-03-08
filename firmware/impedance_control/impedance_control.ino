#include <SimpleKalmanFilter.h>
#include "AS5600.h"

// Impedance control parameters
float K_initial = 6.5;     // Initial stiffness (N/degree)
float B_initial = 2.25;     // Initial damping (N·s/degree)
float I = 451.27 * 1e-9;   // Inertia (kg * m²), obtained from SolidWorks

// Desired values
float desired_angle = 90.0;      // Desired inital angle in degrees for home position(fully extended)
float desired_pressure = 200.0;  // Desired pressure in gram-force

// Kalman filter configurations for velocity, acceleration, and pressure
SimpleKalmanFilter kalman_velocity(1, 1, 0.01);       // Kalman filter for velocity
SimpleKalmanFilter kalman_acceleration(1, 1, 0.01);   // Kalman filter for acceleration
SimpleKalmanFilter kalman_pressure(1, 1, 0.01);       // Kalman filter for pressure

// Sampling period target (microseconds) — 10 ms = 100 Hz
const unsigned long CYCLE_PERIOD_US = 10000UL;

// dt in seconds — updated each cycle from actual elapsed time.
// Used by velocity and acceleration estimators for accuracy.
float dt = 0.01f;

// Use default Wire
AS5600 as5600;

// Output pins
#define PIN_OUTPUT_A 5
#define PIN_OUTPUT_B 6

// Pressure sensor calibration
double pressureSensorOffset = 0.0;

void setup() {
    Serial.begin(115200);
    configureEncoder();
    pressureSensorOffset = pressureSensorCalibration();
}

void loop() {
    // Cycle timer — micros() gives 1 µs resolution.
    // Declared static so it persists across calls and tracks the true
    // start of the previous cycle for accurate dt measurement.
    static unsigned long cycle_start_us = micros();

    // Read current angle and pressure from sensors
    float current_angle    = readAngleSensor();
    float current_pressure = readPressureSensor();
    float filtered_pressure = kalman_pressure.updateEstimate(current_pressure);

    // Update dt from actual elapsed time since the last cycle started.
    // This feeds the velocity/acceleration estimators with the real sample
    // interval instead of the assumed 10 ms, keeping kinematics accurate
    // even when Serial I/O or sensor reads vary in duration.
    dt = constrain((micros() - cycle_start_us) / 1e6f, 0.001f, 0.05f);

    // Read desired angle from Python pipeline over serial.
    // Falls back to last received value if no new data this cycle.
    desired_angle = readDesiredAngle();

    // Real-time adjustment of stiffness and damping (variable impedance control)
    float K = (abs(desired_angle - current_angle) > 5.0) ? K_initial : K_initial * 0.5;
    float B = (abs(kalman_velocity.updateEstimate(0)) > 0.1) ? B_initial : B_initial * 0.8;

    // Calculate position and pressure errors
    float position_error = desired_angle - current_angle;
    float pressure_error = desired_pressure - filtered_pressure;

    // Estimate angular velocity using the Kalman filter
    static float previous_angle = current_angle;
    float angular_velocity = (current_angle - previous_angle) / dt;
    float filtered_velocity = kalman_velocity.updateEstimate(angular_velocity);
    previous_angle = current_angle;

    // Estimate angular acceleration using the Kalman filter
    static float previous_velocity = filtered_velocity;
    float acceleration = (filtered_velocity - previous_velocity) / dt;
    float filtered_acceleration = kalman_acceleration.updateEstimate(acceleration);
    previous_velocity = filtered_velocity;

    // Impedance control law: τ = K(θd−θ) + B(−θ̇) + M(−θ̈) − Fc
    float stiffness_force = K * position_error;
    float damping_force   = B * (-filtered_velocity);
    float inertia_force   = I * (-filtered_acceleration);
    float control_torque  = constrain(
        (stiffness_force + damping_force + inertia_force - filtered_pressure),
        -255, 255
    );

    applyTorque(control_torque);

    // Send status back to Python pipeline (SerialBridge.read_status()).
    // Format: desired_angle, current_angle, pressure, torque, velocity, acceleration
    sendStatus(desired_angle, current_angle, filtered_pressure,
               control_torque, filtered_velocity, filtered_acceleration);

    // Dynamic delay — wait only the time remaining in the 10 ms period.
    // If the loop body took 2 ms, we wait 8 ms.
    // If it overran 10 ms (e.g. Serial stall), skip delay and continue.
    unsigned long elapsed_us = micros() - cycle_start_us;
    if (elapsed_us < CYCLE_PERIOD_US) {
        delayMicroseconds(CYCLE_PERIOD_US - elapsed_us);
    }

    // Reset cycle timer AFTER the delay so the next dt is exact.
    cycle_start_us = micros();
}

// Function to configure the encoder
void configureEncoder() {
    Serial.println(__FILE__);
    Serial.print("AS5600_LIB_VERSION: ");
    Serial.println(AS5600_LIB_VERSION);

    Wire.begin();
    as5600.begin(2);  // Set direction pin
    as5600.setDirection(AS5600_CLOCK_WISE);  // Set direction

    Serial.println(as5600.getAddress());
    Serial.print("Connect: ");
    Serial.println(as5600.isConnected());

    delay(1000);
}

// Function to read the angle sensor
float readAngleSensor() {
    return as5600.rawAngle() * AS5600_RAW_TO_DEGREES;
}

// Function to read the pressure sensor and apply Kalman filter
float readPressureSensor() {
    float voltage = fmap((analogRead(A0) - pressureSensorOffset), 0.0, 1023.0, 0.0, 5.0);
    float pressure = ((100 * voltage) / 0.165); // Pressure in Newtons
    return (pressure < 0) ? 0 : pressure;
}

// Calibration function for the pressure sensor
float pressureSensorCalibration() {
    float sum = 0;
    int samples = 30;
    for (int i = 0; i < samples; i++) {
        sum += analogRead(A0);
        delay(100);
    }
    Serial.print("Calibration done: ");
    Serial.println(sum / samples);
    return sum / samples;
}

// Mapping function for pressure sensor calibration
float fmap(float x, float a, float b, float c, float d) {
    return x / (b - a) * (d - c) + c;
}

// Function to apply torque to the actuator
void applyTorque(float torque) {
    if (torque >= 0) {
        // Move forward
        analogWrite(PIN_OUTPUT_A, abs(torque));  // Set motor forward with computed speed
        analogWrite(PIN_OUTPUT_B, 0);            // Stop reverse direction
    } else {
        // Move backward
        analogWrite(PIN_OUTPUT_A, 0);            // Stop forward direction
        analogWrite(PIN_OUTPUT_B, abs(torque));  // Set motor reverse with computed speed
    }
}


// =============================================================================
// SERIAL COMMUNICATION — swap these two functions to change transport layer
// =============================================================================

/*
 * readDesiredAngle()
 * Reads target angle from pipeline.py. Holds last value between updates.
 * Format: "<float>\n"  e.g. "142.50\n"
 * To revert to potentiometer: uncomment the analogRead line.
 */
float readDesiredAngle() {
    static float last_desired = desired_angle;
    if (Serial.available() > 0) {
        String line = Serial.readStringUntil('\n');
        line.trim();
        float received = line.toFloat();
        if (received >= 0.0 && received <= 360.0) {
            last_desired = received;
        }
    }
    // Potentiometer fallback — uncomment to use manual control:
    // last_desired = map(analogRead(A1), 0, 1023, 70, 288);
    return last_desired;
}

/*
 * sendStatus()
 * Sends controller state to Python pipeline each cycle via serial.
 * CSV format: desired_angle, current_angle, pressure, torque, velocity, accel
 * Match with SerialBridge.read_status() in predict.py if you add fields.
 */
void sendStatus(float d_angle, float c_angle, float pressure,
                float torque,  float velocity, float accel) {
    Serial.print(d_angle,  2); Serial.print(",");
    Serial.print(c_angle,  2); Serial.print(",");
    Serial.print(pressure, 2); Serial.print(",");
    Serial.print(torque,   2); Serial.print(",");
    Serial.print(velocity, 4); Serial.print(",");
    Serial.println(accel,  4);
}
