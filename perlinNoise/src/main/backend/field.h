
#ifndef FIELD_H
#define FIELD_H

typedef float pixel_t;

int createField(pixel_t *fieldOut, long seed, unsigned pixelWidth, long x, long y, unsigned octaves);

pixel_t *fieldAlloc(unsigned pixelWidth);

void fieldFree(pixel_t *field);

#endif
