xcrun metal -o shader.air -c shader.metal
xcrun metallib -o default.metallib shader.air
rm shader.air

