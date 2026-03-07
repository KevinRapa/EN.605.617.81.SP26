
#ifndef PERLIN_H
#define PERLIN_H

static const int FAILURE = 1;
static const int SUCCESS = 0;

static const unsigned FIELD_DIM = 8;
static const unsigned CHUNK_DIM = 32;
static const unsigned OCTAVES = 4;

typedef float pixel_t;

int createField(pixel_t *fieldOut, long seed, int x, int y);

long createFieldSeed(long worldSeed, int x, int y);

pixel_t *fieldAlloc();

void fieldFree(pixel_t *field);

#endif
