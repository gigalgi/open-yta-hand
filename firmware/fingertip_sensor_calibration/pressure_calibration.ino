
double pressureSensorOffset = 0.0;
// the setup routine runs once when you press reset:
void setup() {
  // initialize serial communication at 9600 bits per second:
  Serial.begin(9600);

  pressureSensorOffset = pressureSensorCalibration();
}

// the loop routine runs over and over again forever:
void loop() {
  // read the input on analog pin 0:
  float sensorValue = fmap((analogRead(A0)-pressureSensorOffset),0.0,1023.0,0.0,5.0);
  float pressure = ((100*sensorValue)/0.0375); //pressure in newtons
  // print out the value you read:
  Serial.println(pressure);
  delay(1);  // delay in between reads for stability
}

float fmap(float x, float a, float b, float c, float d)
{
      float f=x/(b-a)*(d-c)+c;
      return f;
}

float pressureSensorCalibration()
{
    float aux = 0;
    float samples = 30;
    for(int i = 0; i <= samples ; i++)
    {
        aux += analogRead(A0);
        delay(100);
    }
    Serial.print("done");
    Serial.println(aux/samples);
    return (aux/samples);
}
