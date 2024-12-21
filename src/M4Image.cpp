#include "M4Image.h"
#include <array>
#include <map>
#include <optional>
#include <memory>
#include <stdlib.h>
#include <limits.h>
#include <scope_guard.hpp>

#include <mango/image/surface.hpp>
#include <mango/image/decoder.hpp>
#include <mango/image/quantize.hpp>
#include <pixman.h>

// done to ensure constant initialization and avoid static initialization order fiasco
static constexpr M4Image::Allocator DEFAULT_ALLOCATOR;
M4Image::Allocator M4Image::allocator = DEFAULT_ALLOCATOR;

struct MallocDeleter {
    void operator()(void* block) {
        M4Image::allocator.freeSafe(block);
    }
};

typedef std::array<unsigned char, USHRT_MAX + 1> UNPREMULTIPLIER_ARRAY;

constexpr UNPREMULTIPLIER_ARRAY createChannelUnpremultiplierArray() {
    UNPREMULTIPLIER_ARRAY channelUnpremultiplierArray = {};

    // note: the alpha, divided by two, is added to the channel
    // so the channel is scaled instead of stripped (it works out to rounding the number, instead of flooring)
    // alpha starts at one, since if it's zero the colour is invisible anyway (and thus would be a divide by zero)
    const size_t DIVIDE_BY_TWO = 1;

    unsigned short tmp = 0;

    for (unsigned short channel = 0; channel <= UCHAR_MAX; channel++) {
        for (unsigned short alpha = 1; alpha <= UCHAR_MAX; alpha++) {
            tmp = ((channel * UCHAR_MAX) + (alpha >> DIVIDE_BY_TWO)) / alpha;
            channelUnpremultiplierArray[(channel << CHAR_BIT) | alpha] = __min(tmp, UCHAR_MAX);
        }
    }
    return channelUnpremultiplierArray;
}

// aligned to nearest 64 bytes so it is on cache lines
// note: there is a specific IntelliSense error that it only shows for a std::array of size 65536 bytes
// this is an IntelliSense bug, it compiles correctly: https://github.com/microsoft/vscode-cpptools/issues/5833
alignas(64) static constexpr UNPREMULTIPLIER_ARRAY CHANNEL_UNPREMULTIPLIER_ARRAY = createChannelUnpremultiplierArray();

#define UNPREMULTIPLY_CHANNEL(channel, alpha) (CHANNEL_UNPREMULTIPLIER_ARRAY[((channel) << CHAR_BIT) | (alpha)])

void convertColors(
    M4Image::Color32* colorPointer,
    size_t width,
    size_t height,
    size_t stride,
    M4Image::COLOR_FORMAT colorFormat,
    unsigned char* imagePointer,
    bool unpremultiply
) {
    const size_t COLOR_CHANNEL_LUMINANCE = 2;
    const size_t COLOR_CHANNEL_ALPHA = 3;

    switch (colorFormat) {
        case M4Image::COLOR_FORMAT::L:
        case M4Image::COLOR_FORMAT::A:
        {
            size_t colorChannel = (colorFormat == M4Image::COLOR_FORMAT::L)
                ? COLOR_CHANNEL_LUMINANCE
                : COLOR_CHANNEL_ALPHA;

            unsigned char* convertedPointer = imagePointer;

            for (size_t i = 0; i < height; i++) {
                for (size_t j = 0; j < width; j++) {
                    *convertedPointer++ = colorPointer++->channels[colorChannel];
                }

                imagePointer += stride;
                convertedPointer = imagePointer;
            }
        }
        return;
        case M4Image::COLOR_FORMAT::XXXL:
        {
            const size_t CONVERTED_CHANNEL_LUMINANCE = 3;

            M4Image::Color32* convertedPointer = (M4Image::Color32*)imagePointer;

            for (size_t i = 0; i < height; i++) {
                for (size_t j = 0; j < width; j++) {
                    convertedPointer++->channels[CONVERTED_CHANNEL_LUMINANCE] = colorPointer++->channels[COLOR_CHANNEL_LUMINANCE];
                }

                imagePointer += stride;
                convertedPointer = (M4Image::Color32*)imagePointer;
            }
        }
        return;
    }

    size_t convertedChannelLuminance = colorFormat != M4Image::COLOR_FORMAT::LA;
    size_t convertedChannelAlpha = colorFormat == M4Image::COLOR_FORMAT::LA;

    M4Image::Color16* convertedPointer = (M4Image::Color16*)imagePointer;

    if (unpremultiply) {
        for (size_t i = 0; i < height; i++) {
            for (size_t j = 0; j < width; j++) {
                unsigned char &alpha = convertedPointer->channels[convertedChannelAlpha];
                alpha = colorPointer->channels[COLOR_CHANNEL_ALPHA];

                if (alpha) {
                    convertedPointer->channels[convertedChannelLuminance] = UNPREMULTIPLY_CHANNEL(colorPointer->channels[COLOR_CHANNEL_LUMINANCE], alpha);
                }

                colorPointer++;
                convertedPointer++;
            }

            imagePointer += stride;
            convertedPointer = (M4Image::Color16*)imagePointer;
        }
        return;
    }

    for (size_t i = 0; i < height; i++) {
        for (size_t j = 0; j < width; j++) {
            M4Image::Color32 &color = *colorPointer;
            M4Image::Color16 &converted = *convertedPointer;

            converted.channels[convertedChannelLuminance] = color.channels[COLOR_CHANNEL_LUMINANCE];
            converted.channels[convertedChannelAlpha] = color.channels[COLOR_CHANNEL_ALPHA];

            colorPointer++;
            convertedPointer++;
        }

        imagePointer += stride;
        convertedPointer = (M4Image::Color16*)imagePointer;
    }
}

void unpremultiplyColors(
    M4Image::Color32* colorPointer,
    size_t width,
    size_t height,
    size_t stride
) {
    const size_t CHANNEL_ALPHA = 3;

    unsigned char* rowPointer = (unsigned char*)colorPointer;

    for (size_t i = 0; i < height; i++) {
        for (size_t j = 0; j < width; j++) {
            const unsigned char &ALPHA = colorPointer->channels[CHANNEL_ALPHA];

            if (ALPHA) {
                colorPointer->channels[0] = UNPREMULTIPLY_CHANNEL(colorPointer->channels[0], ALPHA);
                colorPointer->channels[1] = UNPREMULTIPLY_CHANNEL(colorPointer->channels[1], ALPHA);
                colorPointer->channels[2] = UNPREMULTIPLY_CHANNEL(colorPointer->channels[2], ALPHA);
            }

            colorPointer++;
        }

        rowPointer += stride;
        colorPointer = (M4Image::Color32*)rowPointer;
    }
}

// the mango::core::MemoryStream class does not allow using a custom allocator
// so we must implement our own
class AllocatorStream : public mango::Stream {
    private:
    struct State {
        mango::u64 capacity = 0;
        mango::u64 size = 0;
        mango::u64 offset = 0;
        mango::u8* data = nullptr;
    };

    State state = {};

    public:
    virtual ~AllocatorStream();
    mango::u64 capacity() const;
    mango::u64 size() const;
    mango::u64 offset() const;
    mango::u8* acquire();
    void reserve(mango::u64 offset);
    void seek(mango::s64 distance, SeekMode mode);
    void read(void* dest, mango::u64 size);
    void write(const void* data, mango::u64 size);
};

AllocatorStream::~AllocatorStream() {
    // note that for acquire to work this cannot happen in State's destructor
    M4Image::allocator.freeSafe(state.data);
}

mango::u64 AllocatorStream::capacity() const {
    return state.capacity;
}

mango::u64 AllocatorStream::size() const {
    return state.size;
}

mango::u64 AllocatorStream::offset() const {
    return state.offset;
}

mango::u8* AllocatorStream::acquire() {
    mango::u8* data = state.data;
    state = {};
    return data;
}

void AllocatorStream::reserve(mango::u64 offset) {
    mango::u64 capacity = state.capacity;

    if (offset <= capacity) {
        return;
    }

    if (!capacity) {
        // initial capacity is 128 bytes
        // written as 64 here because we will immediately multiply this by two
        // this number chosen because it is probably the smallest reasonable size for an image
        // (it is the exact size of a 2x2 pixel white PNG)
        // must be a power of two
        const mango::u64 INITIAL_CAPACITY = 64;

        capacity = INITIAL_CAPACITY;
    }

    // find power of two greater than or equal to the offset
    do {
        capacity *= 2;
    } while (offset > capacity);

    M4Image::allocator.reAllocSafe(state.data, capacity);
    state.capacity = capacity;
}

void AllocatorStream::seek(mango::s64 distance, SeekMode mode) {
    switch (mode) {
        case BEGIN:
        state.offset = distance;
        break;
        case CURRENT:
        state.offset += distance;
        break;
        case END:
        state.offset = state.size - distance;
    }
}

void AllocatorStream::read(void* dest, mango::u64 size) {
    if (!size) {
        return;
    }

    mango::u64 offset = state.offset + size;

    // ensure we didn't overflow
    if (offset < state.offset) {
        MANGO_EXCEPTION("[AllocatorStream] Size too large.");
    }

    // ensure we don't read past the end
    if (offset > state.size) {
        MANGO_EXCEPTION("[AllocatorStream] Reading past end of data.");
    }

    // we have verified that size is less than the size of the data
    // so it is safe to use as the source size
    if (memcpy_s(dest, size, state.data + state.offset, size)) {
        MANGO_EXCEPTION("[AllocatorStream] Failed to copy memory.");
    }

    state.offset = offset;
}

void AllocatorStream::write(const void* data, mango::u64 size) {
    if (!size) {
        return;
    }

    mango::u64 offset = state.offset + size;

    // ensure we didn't overflow
    if (offset < state.offset) {
        MANGO_EXCEPTION("[AllocatorStream] Size too large.");
    }

    // ensure we don't write past the end
    reserve(offset);

    // zero seeked over data
    if (state.offset > state.size) {
        memset(state.data + state.size, 0, state.offset - state.size);
    }

    if (memcpy_s(state.data + state.offset, state.capacity - state.offset, data, size)) {
        MANGO_EXCEPTION("[AllocatorStream] Failed to copy memory.");
    }

    // set the size
    if (offset > state.size) {
        state.size = offset;
    }

    state.offset = offset;
}

typedef std::map<M4Image::COLOR_FORMAT, mango::image::Format> FORMAT_MAP;

static const FORMAT_MAP SURFACE_FORMAT_MAP = {
    {M4Image::COLOR_FORMAT::RGBA, mango::image::Format(32, mango::image::Format::UNORM, mango::image::Format::RGBA, 8, 8, 8, 8)},
    {M4Image::COLOR_FORMAT::RGBX, mango::image::Format(32, mango::image::Format::UNORM, mango::image::Format::RGBA, 8, 8, 8, 0)},
    {M4Image::COLOR_FORMAT::BGRA, mango::image::Format(32, mango::image::Format::UNORM, mango::image::Format::BGRA, 8, 8, 8, 8)},
    {M4Image::COLOR_FORMAT::BGRX, mango::image::Format(32, mango::image::Format::UNORM, mango::image::Format::BGRA, 8, 8, 8, 0)},
    {M4Image::COLOR_FORMAT::RGB, mango::image::Format(24, mango::image::Format::UNORM, mango::image::Format::RGB, 8, 8, 8, 0)},
    {M4Image::COLOR_FORMAT::BGR, mango::image::Format(24, mango::image::Format::UNORM, mango::image::Format::BGR, 8, 8, 8, 0)},
    {M4Image::COLOR_FORMAT::LA, mango::image::LuminanceFormat(16, 0x000000FF, 0x0000FF00)},
    {M4Image::COLOR_FORMAT::AL, mango::image::LuminanceFormat(16, 0x0000FF00, 0x000000FF)},
    {M4Image::COLOR_FORMAT::L, mango::image::LuminanceFormat(8, 0x000000FF, 0x00000000)},
    {M4Image::COLOR_FORMAT::A, mango::image::Format(8, mango::image::Format::UNORM, mango::image::Format::A, 8, 0, 0, 0)},
    {M4Image::COLOR_FORMAT::XXXL, mango::image::LuminanceFormat(32, 0xFF000000, 0x00000000)},
    {M4Image::COLOR_FORMAT::XXLA, mango::image::LuminanceFormat(32, 0x00FF0000, 0xFF000000)},
    {M4Image::COLOR_FORMAT::XXLX, mango::image::LuminanceFormat(32, 0x00FF0000, 0x00000000)}
};

// formats used internally for resizing
static const mango::image::Format &SURFACE_FORMAT_RGBA = SURFACE_FORMAT_MAP.at(M4Image::COLOR_FORMAT::RGBA);
static const mango::image::Format &SURFACE_FORMAT_BGRA = SURFACE_FORMAT_MAP.at(M4Image::COLOR_FORMAT::BGRA);
static const mango::image::Format &SURFACE_FORMAT_XXLA = SURFACE_FORMAT_MAP.at(M4Image::COLOR_FORMAT::XXLA);
static const mango::image::Format &SURFACE_FORMAT_XXLX = SURFACE_FORMAT_MAP.at(M4Image::COLOR_FORMAT::XXLX);


// resizes are always done in 32-bit formats
// mango has fast paths for decoding RGBA and BGRA, so these are mostly used
// luminance formats are first decoded in RGBA, then blitted to an equivalent 32-bit format
// they should have alpha if the destination does and shouldn't if it doesn't
// to avoid an otherwise unnecessary allocation during resizing
static const FORMAT_MAP RESIZE_SURFACE_FORMAT_MAP = {
    {M4Image::COLOR_FORMAT::RGBA, SURFACE_FORMAT_RGBA},
    {M4Image::COLOR_FORMAT::RGBX, SURFACE_FORMAT_RGBA},
    {M4Image::COLOR_FORMAT::BGRA, SURFACE_FORMAT_BGRA},
    {M4Image::COLOR_FORMAT::BGRX, SURFACE_FORMAT_BGRA},
    {M4Image::COLOR_FORMAT::RGB, SURFACE_FORMAT_RGBA},
    {M4Image::COLOR_FORMAT::BGR, SURFACE_FORMAT_BGRA},
    {M4Image::COLOR_FORMAT::LA, SURFACE_FORMAT_XXLA},
    {M4Image::COLOR_FORMAT::AL, SURFACE_FORMAT_XXLA},
    {M4Image::COLOR_FORMAT::L, SURFACE_FORMAT_XXLX},
    {M4Image::COLOR_FORMAT::A, SURFACE_FORMAT_RGBA},
    {M4Image::COLOR_FORMAT::XXXL, SURFACE_FORMAT_XXLX},
    {M4Image::COLOR_FORMAT::XXLA, SURFACE_FORMAT_XXLA},
    {M4Image::COLOR_FORMAT::XXLX, SURFACE_FORMAT_XXLX}
};

void blitSurfaceImage(
    const mango::image::Surface &inputSurface,
    mango::image::Surface &outputSurface,
    bool linear = false,
    bool resize = false
) {
    // if we are resizing, then resizeImage will expect the image to be linear
    // if we're dealing with luminance (immediately after this) it will need to be linear as well
    // we don't want to linearize the image for LuminanceBitmap then undo it then redo it again
    // so linearize the image here and keep it that way for the resize
    if (!linear && resize) {
        mango::image::srgbToLinear(inputSurface);
        linear = true;
    }

    std::optional<mango::image::LuminanceBitmap> luminanceBitmapOptional = std::nullopt;

    if (!inputSurface.format.isLuminance() && outputSurface.format.isLuminance()) {
        luminanceBitmapOptional.emplace(inputSurface, outputSurface.format.isAlpha(), linear);
    }

    const mango::image::Surface &SOURCE_SURFACE = luminanceBitmapOptional.has_value() ? luminanceBitmapOptional.value() : inputSurface;

    // if we're forced to do a blit because they don't match, do it
    if (SOURCE_SURFACE.format != outputSurface.format || SOURCE_SURFACE.stride != outputSurface.stride) {
        outputSurface.blit(0, 0, SOURCE_SURFACE);
        return;
    }

    // if we're direct and the image pointers match, they are already equal so copying is unnecessary
    if (SOURCE_SURFACE.image == outputSurface.image) {
        return;
    }

    // if we can avoid a blit and do a direct memory copy, do that instead
    // (it is assumed the caller has ensured the width/height match)
    if (memcpy_s(outputSurface.image, outputSurface.stride * (size_t)outputSurface.height, SOURCE_SURFACE.image, SOURCE_SURFACE.stride * (size_t)SOURCE_SURFACE.height)) {
        throw std::runtime_error("Failed to Copy Memory");
    }
}

void decodeSurfaceImage(
    const mango::image::Surface &luminanceSurface,
    mango::image::Surface &surface,
    mango::image::ImageDecoder &imageDecoder,
    bool linear = false,
    bool resize = false
) {
    // uncomment the second argument to disable multithreading for testing purposes
    mango::image::ImageDecodeStatus status = imageDecoder.decode(luminanceSurface/*, {nullptr, true, false}*/);

    // status is false if decoding the image failed
    if (!status) {
        MANGO_EXCEPTION("[INFO] {}", status.info);
    }

    // for grayscale images we may need to blit them to a luminance format
    blitSurfaceImage(luminanceSurface, surface, linear, resize);
}

unsigned char* encodeSurfaceImage(
    const mango::image::Surface &surface,
    const char* extension,
    size_t &size,
    float quality = 0.90f
) {
    MAKE_SCOPE_EXIT(sizeScopeExit) {
        size = 0;
    };

    AllocatorStream allocatorStream;
    mango::image::ImageEncodeStatus status = surface.save(allocatorStream, extension, { {}, {}, quality });

    if (!status) {
        MANGO_EXCEPTION("[INFO] {}", status.info);
    }

    size = allocatorStream.size();
    unsigned char* bits = allocatorStream.acquire();

    // can technically happen, but probably never will
    if (!bits) {
        throw std::bad_alloc();
    }

    sizeScopeExit.dismiss();
    return bits;
}

pixman_image_t* createImageBits(pixman_format_code_t format, int width, int height, uint32_t* bits, int stride) {
    pixman_image_t* image = pixman_image_create_bits(
        format,
        width, height,
        bits,
        stride
    );

    if (!image) {
        throw std::bad_alloc();
    }
    return image;
}

// if we need to premultiply the colours
// then we use a mask (from which only the alpha channel is used)
// on the surface (which we mark as RGBX, so its alpha is ignored)
// this has the effect of multiplying RGB * Alpha, which
// is the exact operation we need
// (doing it this way is significantly more performant than manually doing it ourselves)
// both use the same underlying data, so no memory is allocated
// they are just seperate views into the same memory
// note that we need the alpha channel for the actual resize
// so we use maskImage as the destination, rather than writing over sourceImage
// therefore the sourceImage (which is RGBX) is only used for this premultiply step
// and the maskImage is what is what we actually transform later on
// if you're totally lost on why this is needed for resizing, then
// see: https://www.realtimerendering.com/blog/gpus-prefer-premultiplication/
pixman_image_t* premultiplyImage(pixman_image_t* image) {
    int width = pixman_image_get_width(image);
    int height = pixman_image_get_height(image);

    pixman_image_t* resultImage = createImageBits(
        PIXMAN_a8r8g8b8,
        width, height,
        pixman_image_get_data(image),
        pixman_image_get_stride(image)
    );

    MAKE_SCOPE_EXIT(resultImageScopeExit) {
        if (!M4Image::unrefImage(resultImage)) {
            throw std::runtime_error("Failed to Unref Image");
        }
    };

    pixman_image_composite(
        PIXMAN_OP_SRC,
        image, resultImage, resultImage,
        0, 0, 0, 0, 0, 0,
        width, height
    );

    resultImageScopeExit.dismiss();
    return resultImage;
}

// Pixman wants a scale (like a percentage to resize by,) not a pixel size
// so here we create that
void setTransform(pixman_image_t* maskImage, const mango::image::Surface &surface, int width, int height) {
    // this is initialized by pixman_transform_init_identity
    pixman_transform_t transform;
    pixman_transform_init_identity(&transform);

    double sx = (double)surface.width / width;
    double sy = (double)surface.height / height;

    if (!pixman_transform_scale(&transform, NULL, pixman_double_to_fixed(sx), pixman_double_to_fixed(sy))) {
        throw std::runtime_error("Failed to Scale Transform");
    }

    if (!pixman_image_set_transform(maskImage, &transform)) {
        throw std::runtime_error("Failed to Set Image Transform");
    }
}

void resizeImage(
    mango::image::Surface &surface,
    int width,
    int height,
    size_t stride,
    M4Image::COLOR_FORMAT colorFormat,
    unsigned char* imagePointer,
    bool linear = false,
    bool premultiplied = false
) {
    // for formats that are only alpha or only luminance, use alpha for source format
    pixman_format_code_t sourceFormat = (colorFormat == M4Image::COLOR_FORMAT::L
        || colorFormat == M4Image::COLOR_FORMAT::A
        || colorFormat == M4Image::COLOR_FORMAT::XXXL
        || colorFormat == M4Image::COLOR_FORMAT::XXLX)

        ? PIXMAN_a8r8g8b8
        : PIXMAN_x8r8g8b8;

    pixman_image_t* sourceImage = createImageBits(
        sourceFormat,
        surface.width, surface.height,
        (uint32_t*)surface.image,
        (int)surface.stride
    );

    SCOPE_EXIT {
        if (!M4Image::unrefImage(sourceImage)) {
            throw std::runtime_error("Failed to Unref Image");
        }
    };

    const mango::image::Format &DESTINATION_SURFACE_FORMAT = SURFACE_FORMAT_MAP.at(colorFormat);
    const mango::image::Color &DESTINATION_SURFACE_FORMAT_SIZE = DESTINATION_SURFACE_FORMAT.size;

    // we should only care about premultiplying if:
    // -the image format isn't already premultiplied (then it's the caller's problem)
    // -the source format is PIXMAN_x8r8g8b8 (indicating we are meant to use it with maskImage)
    // -the destination format has alpha (because otherwise the colours will be unaffected by alpha)
    // -the destination format has RGB channels (because otherwise the colour data will be thrown out anyway)
    // we don't care about if the surface has alpha here
    // the source format will be PIXMAN_x8r8g8b8 if it does/it matters
    bool unpremultiply = !premultiplied
        && sourceFormat == PIXMAN_x8r8g8b8
        && DESTINATION_SURFACE_FORMAT.isAlpha()
        && (DESTINATION_SURFACE_FORMAT_SIZE.r || DESTINATION_SURFACE_FORMAT_SIZE.g || DESTINATION_SURFACE_FORMAT_SIZE.b);

    // premultiply, only if we'll undo it later, and if the original image wasn't already premultiplied
    pixman_image_t* maskImage = unpremultiply
        ? premultiplyImage(sourceImage)
        : sourceImage;

    SCOPE_EXIT {
        if (maskImage != sourceImage) {
            if (!M4Image::unrefImage(maskImage)) {
                throw std::runtime_error("Failed to Unref Image");
            }
        }
    };

    setTransform(maskImage, surface, width, height);

    if (!pixman_image_set_filter(maskImage, PIXMAN_FILTER_BILINEAR, NULL, 0)) {
        throw std::runtime_error("Failed to Set Filter");
    }

    // setting the repeat mode to pad prevents some semi-transparent lines at the edge of the image
    // (because the image is interpreted as being in the middle of a transparent void of pixels otherwise)
    pixman_image_set_repeat(maskImage, PIXMAN_REPEAT_PAD);

    // create the resize bits
    // the resize is always done to BGRA format
    // this way, we can easily unpremultiply and delinearize the result
    // before going to the destination format
    typedef std::unique_ptr<unsigned char[], MallocDeleter> BITS_POINTER;

    // these formats are not 32-bit so will need a larger buffer during the resize
    bool convert = colorFormat == M4Image::COLOR_FORMAT::LA
        || colorFormat == M4Image::COLOR_FORMAT::AL
        || colorFormat == M4Image::COLOR_FORMAT::L
        || colorFormat == M4Image::COLOR_FORMAT::A;

    size_t resizedBitsStride = stride;

    BITS_POINTER resizedBitsPointer = 0;
    unsigned char* resizedBits = imagePointer;

    // if we can write the colours then just swap them in the same space
    // that is fine, we don't need to allocate a new buffer
    // but if the bits per pixel don't match, we can't do that as we'll write into the next colour
    // or, not have enough space
    // and if the destination doesn't have alpha, we shouldn't touch the alpha channel of that buffer
    // in other words, of the 32-bit destination formats, it must be one of the A formats, not X formats
    // to take advantage of not needing to allocate a new buffer
    if (convert
        || surface.format.bits != DESTINATION_SURFACE_FORMAT.bits
        || surface.format.isAlpha() != DESTINATION_SURFACE_FORMAT.isAlpha()) {
        resizedBitsStride = (size_t)width * (size_t)surface.format.bytes();

        resizedBitsPointer = BITS_POINTER((unsigned char*)M4Image::allocator.mallocSafe(resizedBitsStride * (size_t)height));
        resizedBits = resizedBitsPointer.get();
    }

    pixman_image_t* resizedImage = createImageBits(
        PIXMAN_a8r8g8b8,
        width, height,
        (uint32_t*)resizedBits,
        resizedBitsStride
    );

    SCOPE_EXIT {
        if (!M4Image::unrefImage(resizedImage)) {
            throw std::runtime_error("Failed to Unref Image");
        }
    };

    // the actual resize happens here
    pixman_image_composite(
        PIXMAN_OP_SRC,
        maskImage, NULL, resizedImage,
        0, 0, 0, 0, 0, 0,
        width, height
    );

    const mango::image::Surface RESIZED_SURFACE(
        width, height,
        surface.format, resizedBitsStride,
        resizedBits
    );

    // we need to go to sRGB if we aren't expecting a linear image
    if (!linear) {
        // we need to unpremultiply before going to sRGB
        if (unpremultiply) {
            unpremultiplyColors((M4Image::Color32*)resizedBits, width, height, stride);
            unpremultiply = false;
        }

        mango::image::linearToSRGB(RESIZED_SURFACE);
    }

    // mango is capable of these conversions but it's pretty slow at these
    // so I implemented my own for these specific formats
    // the extra check for XXXL is so we don't allocate a buffer unnecessarily above
    // if the image is linear, here it is unpremultiplied simultaneously while converting it
    if (convert || colorFormat == M4Image::COLOR_FORMAT::XXXL) {
        convertColors((M4Image::Color32*)resizedBits, width, height, stride, colorFormat, imagePointer, unpremultiply);
        return;
    }

    // if the image is linear, then we unpremultiply here
    if (unpremultiply) {
        unpremultiplyColors((M4Image::Color32*)resizedBits, width, height, stride);
    }

    // now we just need to get it into the destination format
    blitSurfaceImage(
        RESIZED_SURFACE,
    
        mango::image::Surface(
            width, height,
            DESTINATION_SURFACE_FORMAT, stride,
            imagePointer
        )
    );
}

void M4Image::getInfo(
    const unsigned char* pointer,
    size_t size,
    const char* extension,
    bool* isAlphaPointer,
    uint32_t* bitsPointer,
    int* widthPointer,
    int* heightPointer,
    bool* linearPointer,
    bool* premultipliedPointer
) {
    if (!pointer) {
        throw std::invalid_argument("pointer must not be zero");
    }

    if (!extension) {
        extension = ".";
    }

    mango::image::ImageDecoder imageDecoder(mango::ConstMemory(pointer, size), extension);

    if (!imageDecoder.isDecoder()) {
        throw std::logic_error("No Decoder");
    }

    mango::image::ImageHeader imageHeader = imageDecoder.header();

    if (isAlphaPointer) {
        // this may return true for images that don't actually have alpha sometimes
        // but it doesn't seem to cause any problems
        // (no better solution afaik)
        *isAlphaPointer = imageHeader.format.isAlpha();
    }

    if (bitsPointer) {
        // mango will decode palettes to RGBA by default
        // but we want the images as 8-bit in that case
        // in all other cases it uses the same bits for the format as the input
        // (as far as I can tell)
        *bitsPointer = imageHeader.palette ? 8 : imageHeader.format.bits;
    }

    if (widthPointer) {
        *widthPointer = imageHeader.width;
    }

    if (heightPointer) {
        *heightPointer = imageHeader.height;
    }

    if (linearPointer) {
        *linearPointer = imageHeader.linear;
    }

    if (premultipliedPointer) {
        *premultipliedPointer = imageHeader.premultiplied;
    }
}

M4Image::M4Image(int width, int height, size_t &stride, COLOR_FORMAT colorFormat, unsigned char* imagePointer) {
    create(width, height, stride, colorFormat, imagePointer);
}

M4Image::M4Image(int width, int height) {
    size_t stride = 0;
    create(width, height, stride);
}

M4Image::~M4Image() {
    destroy();
}

typedef std::unique_ptr<mango::u8[], MallocDeleter> SURFACE_IMAGE_POINTER;

void M4Image::blit(const M4Image &m4Image, bool linear, bool premultiplied) {
    if (!imagePointer || !m4Image.imagePointer) {
        throw Invalid();
    }

    const mango::image::Surface INPUT_SURFACE(
        m4Image.width, m4Image.height,
        SURFACE_FORMAT_MAP.at(m4Image.colorFormat), m4Image.stride,
        m4Image.imagePointer
    );

    bool resize = width != m4Image.width || height != m4Image.height;

    const mango::image::Format &OUTPUT_FORMAT = resize ? RESIZE_SURFACE_FORMAT_MAP.at(colorFormat) : SURFACE_FORMAT_MAP.at(colorFormat);

    size_t outputSurfaceStride = stride;
    SURFACE_IMAGE_POINTER outputSurfaceImagePointer = nullptr;

    if (resize) {
        outputSurfaceStride = (size_t)m4Image.width * (size_t)OUTPUT_FORMAT.bytes();
        outputSurfaceImagePointer = SURFACE_IMAGE_POINTER((mango::u8*)M4Image::allocator.mallocSafe(outputSurfaceStride * (size_t)m4Image.height));
    }

    // the resize is not done here, so the input width and height is used for the output surface
    mango::image::Surface outputSurface(
        m4Image.width, m4Image.height,
        OUTPUT_FORMAT, outputSurfaceStride,
        outputSurfaceImagePointer ? outputSurfaceImagePointer.get() : imagePointer
    );
    
    // if the image is alpha only, linearizing it is pointless
    if (colorFormat == M4Image::COLOR_FORMAT::A) {
        linear = true;
    }

    try {
        blitSurfaceImage(INPUT_SURFACE, outputSurface, linear, resize);
    } catch (mango::Exception) {
        throw std::runtime_error("Failed to Blit Surface Image");
    }

    // for our purposes, if the image is opaque, it is as if the image were premultiplied
    if (resize) {
        resizeImage(
            outputSurface,
            width,
            height,
            stride,
            colorFormat,
            imagePointer,
            linear,
            premultiplied || !INPUT_SURFACE.format.isAlpha()
        );
    }
}

void M4Image::load(const unsigned char* pointer, size_t size, const char* extension, bool &linear, bool &premultiplied) {
    MAKE_SCOPE_EXIT(linearScopeExit) {
        linear = false;
    };

    MAKE_SCOPE_EXIT(premultipliedScopeExit) {
        premultiplied = false;
    };

    if (!imagePointer) {
        throw Invalid();
    }

    if (!pointer) {
        throw std::invalid_argument("pointer must not be zero");
    }

    if (!extension) {
        extension = ".";
    }

    mango::image::ImageDecoder imageDecoder(mango::ConstMemory(pointer, size), extension);

    if (!imageDecoder.isDecoder()) {
        throw std::logic_error("No Decoder");
    }

    mango::image::ImageHeader imageHeader = imageDecoder.header();

    linear = imageHeader.linear;
    premultiplied = imageHeader.premultiplied;
    bool resize = width != imageHeader.width || height != imageHeader.height;

    const mango::image::Format &SURFACE_FORMAT = resize ? RESIZE_SURFACE_FORMAT_MAP.at(colorFormat) : SURFACE_FORMAT_MAP.at(colorFormat);

    size_t surfaceStride = stride;
    SURFACE_IMAGE_POINTER surfaceImagePointer = nullptr;

    // if we're resizing, create a buffer that recieves the originally sized image
    if (resize) {
        surfaceStride = (size_t)imageHeader.width * (size_t)SURFACE_FORMAT.bytes();
        surfaceImagePointer = SURFACE_IMAGE_POINTER((mango::u8*)M4Image::allocator.mallocSafe(surfaceStride * (size_t)imageHeader.height));
    }

    mango::image::Surface surface(
        imageHeader.width, imageHeader.height,
        SURFACE_FORMAT, surfaceStride,
        surfaceImagePointer ? surfaceImagePointer.get() : imagePointer
    );

    // if it's alpha only, there is no point in linearizing so skip it
    // however, the caller should not know about this
    bool resizeLinear = colorFormat == M4Image::COLOR_FORMAT::A ? true : linear;

    // scope for temporary luminance surface stuff
    {
        bool isLuminance = SURFACE_FORMAT.isLuminance();

        // LuminanceBitmap uses RGBA natively, so import to that if the blit format is luminance
        const mango::image::Format &LUMINANCE_SURFACE_FORMAT = isLuminance
            ? SURFACE_FORMAT_RGBA
            : SURFACE_FORMAT;

        size_t luminanceSurfaceStride = surface.stride;
        SURFACE_IMAGE_POINTER luminanceSurfaceImagePointer = nullptr;

        if (isLuminance) {
            luminanceSurfaceStride = (size_t)imageHeader.width * (size_t)LUMINANCE_SURFACE_FORMAT.bytes();

            // if we are dealing with luminance
            // there will always be a LuminanceBitmap intermediary
            // so we can safely use the same memory for
            // both the RGBA and XXLA side of things, as we'll never
            // end up interpreting the same memory as a different format for the blit
            // we should only allocate this if it needs to be a different size
            // like RGBA to L for example
            if (luminanceSurfaceStride != surfaceStride) {
                luminanceSurfaceImagePointer = SURFACE_IMAGE_POINTER((mango::u8*)M4Image::allocator.mallocSafe(luminanceSurfaceStride * (size_t)imageHeader.height));
            }
        }

        try {
            decodeSurfaceImage(
                mango::image::Surface(
                    imageHeader.width, imageHeader.height,
                    LUMINANCE_SURFACE_FORMAT, luminanceSurfaceStride,
                    luminanceSurfaceImagePointer ? luminanceSurfaceImagePointer.get() : surface.image
                ),

                surface,
                imageDecoder,
                resizeLinear,
                resize
            );
        } catch (mango::Exception) {
            throw std::runtime_error("Failed to Decode Surface Image");
        }
    }

    // here we use the same trick where if the image is opaque, we say it's premultiplied
    // however the caller should not get to know this
    if (resize) {
        resizeImage(
            surface,
            width,
            height,
            stride,
            colorFormat,
            imagePointer,
            resizeLinear,
            premultiplied || !imageHeader.format.isAlpha()
        );
    }

    premultipliedScopeExit.dismiss();
    linearScopeExit.dismiss();
}

void M4Image::load(const unsigned char* pointer, size_t size, const char* extension, bool &linear) {
    bool premultiplied = false;
    load(pointer, size, extension, linear, premultiplied);
}

void M4Image::load(const unsigned char* pointer, size_t size, const char* extension) {
    bool linear = false;
    load(pointer, size, extension, linear);
}

unsigned char* M4Image::save(size_t &size, const char* extension, float quality) const {
    MAKE_SCOPE_EXIT(sizeScopeExit) {
        size = 0;
    };

    if (!imagePointer) {
        throw Invalid();
    }

    if (!extension) {
        extension = ".";
    }

    unsigned char* bits = 0;

    try {
        bits = encodeSurfaceImage(
            mango::image::Surface(
                width, height,
                SURFACE_FORMAT_MAP.at(colorFormat), stride,
                imagePointer
            ),

            extension,
            size,
            quality
        );
    } catch (mango::Exception) {
        throw std::runtime_error("Failed to Encode Surface Image");
    }

    sizeScopeExit.dismiss();
    return bits;
}

unsigned char* M4Image::acquire() {
    unsigned char* imagePointer = this->imagePointer;
    this->imagePointer = 0;
    return imagePointer;
}

void M4Image::create(int width, int height, size_t &stride, COLOR_FORMAT colorFormat, unsigned char* imagePointer) {
    if (!width || !height) {
        throw std::invalid_argument("width and height must not be zero");
    }

    if (!stride) {
        stride = (size_t)width * (size_t)SURFACE_FORMAT_MAP.at(colorFormat).bytes();
    }

    if (!imagePointer) {
        imagePointer = (unsigned char*)M4Image::allocator.mallocSafe(stride * (size_t)height);
        owner = true;
    }

    this->width = width;
    this->height = height;
    this->stride = stride;
    this->colorFormat = colorFormat;
    this->imagePointer = imagePointer;
}

void M4Image::destroy() {
    if (owner) {
        M4Image::allocator.freeSafe(imagePointer);
    }
}