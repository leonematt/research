/*
Copyright (C) 2023 Matthew Leon
SPDX-License-Identifier: BSD-3-Clause
*/

#ifndef STREAM_H
#define STREAM_H

#include <string>
#include <iostream>
#include <memory>

using namespace std;

class Stream
{

public:

    shared_ptr<string> buffer;

    virtual ~Stream() {}

    virtual void open() {}
    virtual void close() {}
    virtual void read() {}
    virtual void write() {}
    virtual void seek() {}

};


class BufferedStream : Stream
{

public:

    BufferedStream() 
    {

        open();

    }
    ~BufferedStream() override {}

    void open() override
    {

        buffer = make_shared<string>("");
        cout << "Created the BufferedStream" << endl;
    }

    void close() override
    {

        buffer.reset();
        cout << "Deleted the BufferedStream" << endl;

    }

    void read() override
    {

        cout << "Read the BufferedStream" << endl;

    }

    void write() override
    {

        cout << "Wrote to the BufferedStream" << endl;

    }

    void seek() override
    {

        cout << "Moved location in the BufferedStream" << endl;

    }

};

class EncryptedStream : Stream
{

public:

    shared_ptr<Stream> baseStream;

    EncryptedStream(shared_ptr<Stream> bStream)
    {

        baseStream = bStream;
        cout << "Stream is now encrypted" << endl;

    }
    ~EncryptedStream() override
    {

        if (baseStream.use_count())
            close();

    }

    void close() override
    {

        cout << "Deleting the EncryptedStream" << endl;
        baseStream->close();
        baseStream.reset();

    }

    void write() override
    {

        if (baseStream) {

            cout << "Encrypting The Data" << endl;
            baseStream->write();

        }

    }

};

class CompressedStream : Stream
{

public:

    shared_ptr<Stream> baseStream;

    CompressedStream(shared_ptr<Stream> bStream)
    {

        baseStream = bStream;
        cout << "Stream is now compressed" << endl;

    }
    ~CompressedStream() override
    {

        if (baseStream.use_count())
            close();

    }

    void close() override
    {

        cout << "Deleting the CompressedStream" << endl;
        baseStream->close();

    }

    void write() override
    {

        if (baseStream) {
    
            cout << "Compressing The Data" << endl;
            baseStream->write();

        }

    }

};

#endif // STREAM_H