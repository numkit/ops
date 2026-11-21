// toolboxes/builtin/src/math/random/matlab_mt19937.cpp
//
// MATLAB-canonical MT19937 RNG delegating to keel::random::Mt19937.

#include <keel/core.h>
#include <numkit/ops/matlab_mt19937.hpp>

namespace numkit::ops {

keel::u32 MatlabMT19937::genUint32()
{
    return mt_.next_u32();
}

keel::f64 MatlabMT19937::genRes53()
{
    return mt_.next_f64_open();
}

void MatlabMT19937::getState(keel::u32 out[STATE_SIZE], int &outIndex) const
{
    keel::usize idx = 0;
    mt_.get_state(keel::MutArrayRef<keel::u32>(out, STATE_SIZE), idx);
    outIndex = keel::as<int>(idx);
}

void MatlabMT19937::setState(const keel::u32 in[STATE_SIZE], int inIndex)
{
    const keel::usize idx = (inIndex < 0) ? STATE_SIZE : keel::as<keel::usize>(inIndex);
    mt_.set_state(keel::ArrayRef<keel::u32>(in, STATE_SIZE), idx);
}

} // namespace numkit::ops
