#include <MyoWare.h>
#include <HX711.h>

// Pins:
const int semg_env_pin = A0;

const int loadplate_data_pin = 4;
const int loadplate_clock_pin = 5;

const int interval = 1000;
unsigned long prevMicros = 0;

uint32_t semg_count = 0;
// // Class objects
// Myoware semg
HX711 loadplate;

void setup() {
  // Initialize serial communication
  Serial.begin(115200);
  Serial.println(); 
  Serial.println("Initializing...");

  // // load plate setup
  loadplate.begin(loadplate_data_pin, loadplate_clock_pin);

  loadplate.set_scale(-2552.6528); // Constant found by calibrating according to README in HX711 package
  loadplate.tare(10);
  Serial.println("Starting...");

  delay(3000);
  loadplate.get_units(); //Clear out any previously stored value

  
}

void loop() {
  // Non-blocking delay
  unsigned long currentMicros = micros();
  if(currentMicros - prevMicros >= interval){
    prevMicros = currentMicros;

    // Read semgValue and save as 16-bit unsigned int
    uint16_t semgValue = analogRead(semg_env_pin);

    // Break 16 bits into 2 8-bit packages, with one byte as a header '0xE1'
    uint8_t buf1[3];
    buf1[0] = 0xE1;
    buf1[1] = semgValue & 0xFF; 
    buf1[2] = (semgValue >> 8) & 0xFF;

    // Write to serial port
    Serial.write(buf1, 3);

    // Check if loadplate has value to read
    if(loadplate.is_ready()){
      // Get loadplate value as 32-bit unsigned int
      int16_t forceValue = int(loadplate.get_units());

      // Break the 32-bit semg counter and 32-bit loadplate value into 8-bit packages, with one byte as header '0xF1'
      uint8_t buf2[7];
      buf2[0] = 0xF1;
      buf2[1] = semg_count & 0xFF;
      buf2[2] = (semg_count >> 8) & 0xFF;
      buf2[3] = (semg_count >> 16) & 0xFF;
      buf2[4] = (semg_count >> 24) & 0xFF;

      buf2[5] = forceValue & 0xFF;
      buf2[6] = (forceValue >> 8) & 0xFF;

      // Write to serial port
      Serial.write(buf2, 7);
    }

    // Increase semg counter (~1 per ms)
    semg_count++;

  }
}
