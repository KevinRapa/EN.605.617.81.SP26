
#ifndef CRC64_H
#define CRC64_H

typedef unsigned long long DWORD64;

extern const DWORD64 crctab64[256];

#define CRC_INIT 0xFFFFFFFFFFFFFFFFULL

/**
 * CRC64 function. Computes a CRC 64 hash from an array of bytes
 *
 * @param[in] data pointer to the buffer
 * @param[in] len length of buffer in bytes
 * @param[in,out] hash in = initial value of the hash; out = returned hash
 */
void GetCRC64(DWORD64 &crc, const unsigned char *cp, unsigned long length);

#endif
