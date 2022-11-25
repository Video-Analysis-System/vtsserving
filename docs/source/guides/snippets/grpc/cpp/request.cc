#include "vtsserving/grpc/v1/service.pb.h"

using vtsserving::grpc::v1::VtsService;
using vtsserving::grpc::v1::NDArray;
using vtsserving::grpc::v1::Request;

std::vector<float> data = {3.5, 2.4, 7.8, 5.1};
std::vector<int> shape = {1, 4};

Request request;
request.set_api_name("classify");

NDArray *ndarray = request.mutable_ndarray();
ndarray->mutable_shape()->Assign(shape.begin(), shape.end());
ndarray->mutable_float_values()->Assign(data.begin(), data.end());
