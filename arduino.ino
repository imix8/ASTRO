#include <Servo.h>

// Servo pins
Servo servo1; // D2
Servo servo2; // D3
Servo servo3; // D4

// Stepper Motor 1
const int ena1 = 5;
const int dir1 = 6;
const int step1 = 7;

// Stepper Motor 2
const int ena2 = 8;
const int dir2 = 9;
const int step2 = 10;

// Simulated 3-bit inputs
byte directionBits[] = {
    0b000, // Turn RIGHT
    0b001, // Turn LEFT
    0b010, // FORWARD
    0b011, // BACKWARD
    0b111  // STOP + servo
};

byte servo1Input[] = {
    0, 1, 0, 1, 0 // Used only when direction == 0b111
};

byte weightBits[] = {
    0, 0, 0, 0, 1 // 0→1 transition triggers servo reset
};

int inputIndex = 0;
int prevWeight = 0;

void setup() {
  // Servo setup
  servo1.attach(2);
  servo2.attach(3);
  servo3.attach(4);
  servo1.write(0);
  servo2.write(0);
  servo3.write(0);

  // Stepper setup
  pinMode(step1, OUTPUT);
  pinMode(dir1, OUTPUT);
  pinMode(ena1, OUTPUT);
  pinMode(step2, OUTPUT);
  pinMode(dir2, OUTPUT);
  pinMode(ena2, OUTPUT);

  digitalWrite(ena1, LOW); // Enable both motors
  digitalWrite(ena2, LOW);

  Serial.begin(9600);
  Serial.println("Robot system with 3-bit input started");
}

void loop() {
  if (Serial.available()) {
    byte command = Serial.read();

    byte direction = directionBits[inputIndex];
    byte servoInput = servo1Input[inputIndex];
    int weight = weightBits[inputIndex];

    switch (command) {
    case 0b000: // Turn RIGHT
      Serial.println("Input 000: Turn RIGHT");
      digitalWrite(dir1, HIGH); // Left motor forward
      digitalWrite(dir2, LOW);  // Right motor backward
      stepBothMotors();
      break;

    case 0b001: // Turn LEFT
      Serial.println("Input 001: Turn LEFT");
      digitalWrite(dir1, LOW);  // Left motor backward
      digitalWrite(dir2, HIGH); // Right motor forward
      stepBothMotors();
      break;

    case 0b010: // FORWARD
      Serial.println("Input 010: Move FORWARD");
      digitalWrite(dir1, HIGH); // Both forward
      digitalWrite(dir2, HIGH);
      stepBothMotors();
      break;

    case 0b011: // BACKWARD
      Serial.println("Input 011: Move BACKWARD");
      digitalWrite(dir1, LOW); // Both backward
      digitalWrite(dir2, LOW);
      stepBothMotors();
      break;

    case 0b111: // STOP + servo action
      Serial.println("Input 111: STOP + Activate Servos");

      delay(500); // Pause motion

      // Servo2 & Servo3 to 90°
      servo2.write(90);
      servo3.write(90);
      delay(500);

      // Servo1 depending on input
      if (servoInput == 0) {
        Serial.println("servo1: rotate LEFT 45°");
        servo1.write(45);
      } else {
        Serial.println("servo1: rotate RIGHT 45°");
        servo1.write(135);
      }

      delay(1000);
      break;

    default:
      Serial.println("Invalid direction input");
      break;
    }

    // Detect weight change from 0 → 1
    if (prevWeight == 0 && weight == 1) {
      Serial.println("Weight detected → Resetting servos to 0°");
      servo1.write(0);
      servo2.write(0);
      servo3.write(0);
    }

    prevWeight = weight;

    inputIndex++;
    if (inputIndex >= sizeof(directionBits)) {
      inputIndex = sizeof(directionBits) - 1; // Stop at last input
    }

    delay(1000);
  }
}

void stepBothMotors() {
  for (int i = 0; i < 200; i++) {
    digitalWrite(step1, HIGH);
    digitalWrite(step2, HIGH);
    delayMicroseconds(500);
    digitalWrite(step1, LOW);
    digitalWrite(step2, LOW);
    delayMicroseconds(500);
  }
}