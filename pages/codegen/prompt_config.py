convert_to_cpp_temp = """

Convert below code to C++:
```
{}
```
Only allow replies to pure C++ code.
"""

example_temp = """

The following code is an example of UDF in velox:

```
#include <velox/expression/VectorFunction.h>
#include <iostream>
#include "udf/Udf.h"

namespace {
using namespace facebook::velox;

template <TypeKind Kind>
class PlusConstantFunction : public exec::VectorFunction {
 public:
  explicit PlusConstantFunction(int32_t addition) : addition_(addition) {}

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& /* outputType */,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    using nativeType = typename TypeTraits<Kind>::NativeType;
    VELOX_CHECK_EQ(args.size(), 1);

    auto& arg = args[0];

    // The argument may be flat or constant.
    VELOX_CHECK(arg->isFlatEncoding() || arg->isConstantEncoding());

    BaseVector::ensureWritable(rows, createScalarType<Kind>(), context.pool(), result);

    auto* flatResult = result->asFlatVector<nativeType>();
    auto* rawResult = flatResult->mutableRawValues();

    flatResult->clearNulls(rows);

    if (arg->isConstantEncoding()) {
      auto value = arg->as<ConstantVector<nativeType>>()->valueAt(0);
      rows.applyToSelected([&](auto row) { rawResult[row] = value + addition_; });
    } else {
      auto* rawInput = arg->as<FlatVector<nativeType>>()->rawValues();

      rows.applyToSelected([&](auto row) { rawResult[row] = rawInput[row] + addition_; });
    }
  }

 private:
  const int32_t addition_;
};

static std::vector<std::shared_ptr<exec::FunctionSignature>> integerSignatures() {
  // integer -> integer
  return {exec::FunctionSignatureBuilder().returnType("integer").argumentType("integer").build()};
}

static std::vector<std::shared_ptr<exec::FunctionSignature>> bigintSignatures() {
  // bigint -> bigint
  return {exec::FunctionSignatureBuilder().returnType("bigint").argumentType("bigint").build()};
}

} // namespace

const int kNumMyUdf = 2;
gluten::UdfEntry myUdf[kNumMyUdf] = {{"myudf1", "integer"}, {"myudf2", "bigint"}};

DEFINE_GET_NUM_UDF {
  return kNumMyUdf;
}

DEFINE_GET_UDF_ENTRIES {
  for (auto i = 0; i < kNumMyUdf; ++i) {
    udfEntries[i] = myUdf[i];
  }
}

DEFINE_REGISTER_UDF {
  facebook::velox::exec::registerVectorFunction(
      "myudf1", integerSignatures(), std::make_unique<PlusConstantFunction<facebook::velox::TypeKind::INTEGER>>(5));
  facebook::velox::exec::registerVectorFunction(
      "myudf2", bigintSignatures(), std::make_unique<PlusConstantFunction<facebook::velox::TypeKind::BIGINT>>(5));
  LOG(INFO) << "registered myudf1, myudf2";
}

```

Please refer code in the example and rewrite code I provided into a C++ velox based UDF.
Here is code to convert:
```
"""

generate_search_query = """
Your task is to convert the code I provide into C++ code that complies with the Velox UDF specification. Based on the code I provided,  write 3 the necessary search keywords to gather information from the Velox code or document.

## Rule
- Don't ask questions you already know and velox udf specification
- The main purpose is to find functions, type definitions, etc. that already exist and can be directly used in Velox

Here is code:
```
{}
```

Only respond in the following JSON format:
```
{
"Queries":[
<QUERY 1>,
"<QUERY 2>"
]
}
"""

rag_suffix = """
Some Velox code for reference:
```
{}
```
"""


