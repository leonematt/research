/*
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
*/

#include "stream.h"

int main(void)
{

    // Create the buffered stream
    shared_ptr<Stream> streamPTR((Stream *)new BufferedStream());

    // Add encryption to the stream
    streamPTR.reset((Stream *)new EncryptedStream(streamPTR));

    // Add compression to the stream
    streamPTR.reset((Stream *)new CompressedStream(streamPTR));

    cout << endl << endl;

    // Write to the compressed, encrypted, stream
    streamPTR->write();

    cout << endl << endl;

    return 0;

}