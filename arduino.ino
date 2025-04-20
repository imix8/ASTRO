#include <Servo.h>

// Stepper Motor 1
const int ena1 = 5;
const int dir1 = 6;
const int step1 = 7;

// Stepper Motor 2
const int ena2 = 8;
const int dir2 = 9;
const int step2 = 10;

void setup() {
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
  Serial.println("Stepper motor system started");
}

void loop() {
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');

    if (cmd == "right") {
      Serial.println("Input right");
      digitalWrite(dir1, LOW); // Left motor forward
      digitalWrite(dir2, LOW); // Right motor backward
      stepBothMotors();
    } else if (cmd == "left") {
      Serial.println("Input left");
      digitalWrite(dir1, HIGH); // Left motor backward
      digitalWrite(dir2, HIGH); // Right motor forward
      stepBothMotors();
    } else if (cmd == "forward") {
      Serial.println("Input forward");
      digitalWrite(dir1, LOW); // Both forward
      digitalWrite(dir2, HIGH);
      stepBothMotors();
    } else if (cmd == "backward") {
      Serial.println("Input backwardD");
      digitalWrite(dir1, HIGH); // Both backward
      digitalWrite(dir2, LOW);
      stepBothMotors();
    } else if (cmd == "stop_servo") {
      Serial.println("Input stop_servo");
      delay(500); // Short pause
    } else {
      Serial.println("Invalid direction input");
    }
  }
}

void stepBothMotors() {
  for (int i = 0; i < 200; i++) {
    digitalWrite(step1, HIGH);
    digitalWrite(step2, HIGH);
    delayMicroseconds(1000);
    digitalWrite(step1, LOW);
    digitalWrite(step2, LOW);
    delayMicroseconds(1000);
  }
}
