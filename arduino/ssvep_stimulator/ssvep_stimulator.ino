/*
 * SSVEP LED Stimulator with Serial Control
 *
 * This sketch controls 4 white LEDs flickering at SSVEP frequencies
 * and 4 red LEDs for BCI feedback. Supports both button control
 * and serial commands from Python.
 *
 * Hardware:
 *   - White LEDs: D2 (8.57Hz), D3 (10Hz), D4 (12Hz), D5 (15Hz)
 *   - Red LEDs: D6, D7, D8, D9 (feedback indicators)
 *   - Button: D10 (INPUT_PULLUP, start/stop toggle)
 *
 * Serial Protocol (115200 baud):
 *   Commands (receive):
 *     "START"      - Start LED flickering
 *     "STOP"       - Stop all LEDs
 *     "FEEDBACK:N" - Light red LED N (0-3) as BCI feedback
 *     "CLEAR"      - Turn off all red LEDs
 *
 *   Responses (send):
 *     "OK"         - Command acknowledged
 *     "RUNNING"    - LEDs started
 *     "STOPPED"    - LEDs stopped
 */

// Pin definitions
const int whitePins[] = {2, 3, 4, 5};  // SSVEP flicker LEDs
const int redPins[] = {6, 7, 8, 9};    // Feedback LEDs
const int BUTTON_PIN = 10;
const int NUM_LEDS = 4;

// Flicker frequencies (Hz) and half-periods (microseconds)
const float frequencies[] = {8.57, 10.0, 12.0, 15.0};
unsigned long halfPeriods[] = {58333, 50000, 41667, 33333};

// LED state
bool ledStates[] = {false, false, false, false};
unsigned long nextToggle[] = {0, 0, 0, 0};

// System state
bool isRunning = false;
bool lastButtonState = HIGH;
unsigned long lastDebounceTime = 0;
const unsigned long DEBOUNCE_DELAY = 50;

// Serial command buffer
String commandBuffer = "";

// Feedback state
int activeFeedbackLED = -1;

void setup() {
  // Initialize serial
  Serial.begin(115200);

  // Initialize white LED pins
  for (int i = 0; i < NUM_LEDS; i++) {
    pinMode(whitePins[i], OUTPUT);
    digitalWrite(whitePins[i], LOW);
  }

  // Initialize red LED pins
  for (int i = 0; i < NUM_LEDS; i++) {
    pinMode(redPins[i], OUTPUT);
    digitalWrite(redPins[i], LOW);
  }

  // Initialize button
  pinMode(BUTTON_PIN, INPUT_PULLUP);

  Serial.println("SSVEP Stimulator Ready");
  Serial.println("Commands: START, STOP, FEEDBACK:N, CLEAR");
}

void loop() {
  // Handle serial commands
  handleSerial();

  // Handle button input
  handleButton();

  // Update LED flickering
  if (isRunning) {
    updateFlicker();
  }
}

void handleSerial() {
  while (Serial.available()) {
    char c = Serial.read();

    if (c == '\n' || c == '\r') {
      if (commandBuffer.length() > 0) {
        processCommand(commandBuffer);
        commandBuffer = "";
      }
    } else {
      commandBuffer += c;
    }
  }
}

void processCommand(String cmd) {
  cmd.trim();
  cmd.toUpperCase();

  if (cmd == "START") {
    startStimulation();
    Serial.println("OK");
    Serial.println("RUNNING");
  }
  else if (cmd == "STOP") {
    stopStimulation();
    Serial.println("OK");
    Serial.println("STOPPED");
  }
  else if (cmd.startsWith("FEEDBACK:")) {
    int ledIndex = cmd.substring(9).toInt();
    if (ledIndex >= 0 && ledIndex < NUM_LEDS) {
      showFeedback(ledIndex);
      Serial.println("OK");
    } else {
      Serial.println("ERROR: Invalid LED index");
    }
  }
  else if (cmd == "CLEAR") {
    clearFeedback();
    Serial.println("OK");
  }
  else if (cmd == "STATUS") {
    if (isRunning) {
      Serial.println("RUNNING");
    } else {
      Serial.println("STOPPED");
    }
  }
  else {
    Serial.println("ERROR: Unknown command");
  }
}

void handleButton() {
  int reading = digitalRead(BUTTON_PIN);

  if (reading != lastButtonState) {
    lastDebounceTime = millis();
  }

  if ((millis() - lastDebounceTime) > DEBOUNCE_DELAY) {
    if (reading == LOW && lastButtonState == HIGH) {
      // Button pressed - toggle stimulation
      if (isRunning) {
        stopStimulation();
        Serial.println("STOPPED");
      } else {
        startStimulation();
        Serial.println("RUNNING");
      }
    }
  }

  lastButtonState = reading;
}

void startStimulation() {
  isRunning = true;
  unsigned long now = micros();

  // Initialize toggle times
  for (int i = 0; i < NUM_LEDS; i++) {
    nextToggle[i] = now + halfPeriods[i];
    ledStates[i] = true;
    digitalWrite(whitePins[i], HIGH);
  }
}

void stopStimulation() {
  isRunning = false;

  // Turn off all white LEDs
  for (int i = 0; i < NUM_LEDS; i++) {
    digitalWrite(whitePins[i], LOW);
    ledStates[i] = false;
  }
}

void updateFlicker() {
  unsigned long now = micros();

  for (int i = 0; i < NUM_LEDS; i++) {
    if (now >= nextToggle[i]) {
      // Toggle LED
      ledStates[i] = !ledStates[i];
      digitalWrite(whitePins[i], ledStates[i] ? HIGH : LOW);

      // Schedule next toggle
      nextToggle[i] = now + halfPeriods[i];
    }
  }
}

void showFeedback(int ledIndex) {
  // Clear previous feedback
  clearFeedback();

  // Light up the specified red LED
  if (ledIndex >= 0 && ledIndex < NUM_LEDS) {
    digitalWrite(redPins[ledIndex], HIGH);
    activeFeedbackLED = ledIndex;
  }
}

void clearFeedback() {
  // Turn off all red LEDs
  for (int i = 0; i < NUM_LEDS; i++) {
    digitalWrite(redPins[i], LOW);
  }
  activeFeedbackLED = -1;
}
