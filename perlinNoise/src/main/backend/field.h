
#ifndef FIELD_H
#define FIELD_H

typedef float pixel_t;

int createField(pixel_t *fieldOut, long seed, long x, long y, unsigned octaves);

pixel_t *fieldAlloc();

void fieldFree(pixel_t *field);

#endif
