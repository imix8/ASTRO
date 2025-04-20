#include <Servo.h>

// Stepper Motor 1
const int ena1 = 5;
const int dir1 = 6;
const int step1 = 7;

// Stepper Motor 2
const int ena2 = 8;
const int dir2 = 9;
const int step2 = 10;

// Servo Motors
Servo servo1;
Servo servo2;
Servo servo3;
const int servo1Pin = 2;
const int servo2Pin = 3;
const int servo3Pin = 4;

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

  // Servo setup
  servo1.attach(servo1Pin);
  servo2.attach(servo2Pin);
  servo3.attach(servo3Pin);

  servo1.write(70); // Initial position
  servo2.write(0);  // Initial position
  servo3.write(0);  // Initial position

  Serial.begin(9600);
  Serial.println("Stepper motor system started");
}

void loop() {
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');

    if (cmd == "right") {
      Serial.println("Input right");
      digitalWrite(dir1, LOW);
      digitalWrite(dir2, LOW);
      stepBothMotors();
    } else if (cmd == "left") {
      Serial.println("Input left");
      digitalWrite(dir1, HIGH);
      digitalWrite(dir2, HIGH);
      stepBothMotors();
    } else if (cmd == "forward") {
      Serial.println("Input forward");
      digitalWrite(dir1, LOW);
      digitalWrite(dir2, HIGH);
      stepBothMotors();
    } else if (cmd == "backward") {
      Serial.println("Input backward");
      digitalWrite(dir1, HIGH);
      digitalWrite(dir2, LOW);
      stepBothMotors();
    } else if (cmd == "stop_servo") {
      Serial.println("Input stop_servo");

      // Move servo1 from 70 → 0
      for (int angle = 70; angle >= 0; angle--) {
        servo1.write(angle);
        delay(15);
      }

      // Move servo2 from 0 → 70
      for (int angle = 0; angle <= 70; angle++) {
        servo2.write(angle);
        delay(15);
      }

    } else if (cmd == "sorting1") {
      Serial.println("Input sorting1");
      servo3.write(0);
    } else if (cmd == "sorting2") {
      Serial.println("Input sorting2");
      servo3.write(90);
    } else {
      Serial.println("Invalid direction input");
    }
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