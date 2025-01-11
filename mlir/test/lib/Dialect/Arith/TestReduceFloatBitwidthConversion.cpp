//===- TestReduceFloatBitwdithConversion.cpp ----------------*- c++ -----*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A pass that reduces the bitwidth of Arith floating-point IR.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::arith;

namespace {

class ConstantOpPattern : public OpConversionPattern<ConstantOp> {
  using OpConversionPattern<ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    double val = cast<FloatAttr>(op.getValue()).getValueAsDouble();
    auto newAttr = FloatAttr::get(Float16Type::get(op.getContext()), val);
    rewriter.replaceOpWithNewOp<ConstantOp>(op, newAttr);
    return success();
  }
};

struct TestReduceFloatBitwidthConversionPass
    : public PassWrapper<TestReduceFloatBitwidthConversionPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestReduceFloatBitwidthConversionPass)

  TestReduceFloatBitwidthConversionPass() = default;
  TestReduceFloatBitwidthConversionPass(const TestReduceFloatBitwidthConversionPass &pass)
      : PassWrapper(pass) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
  }
  StringRef getArgument() const final {
    return "test-arith-reduce-float-bitwidth-conversion";
  }
  StringRef getDescription() const final {
    return "Pass that reduces the bitwidth of floating-point ops (dialect conversion)";
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();

    RewritePatternSet patterns(ctx);
    patterns.insert<ConstantOpPattern>(ctx);

    TypeConverter converter;
    converter.addConversion([](Type type) { return type; });
    converter.addConversion([&](Float32Type type) {
      return FloatType::getF16(ctx);
    });

    ConversionTarget target(*ctx);
    target.markUnknownOpDynamicallyLegal([&](Operation *op) {
      return converter.isLegal(op);    
    });

    if (failed(
            applyPartialConversion(getOperation(), target, std::move(patterns)))) {
      getOperation()->emitError() << getArgument() << " failed";
      signalPassFailure();
            }
  }
};
} // namespace

namespace mlir::test {
void registerTestReduceFloatBitwidthConversionPass() {
  PassRegistration<TestReduceFloatBitwidthConversionPass>();
}
} // namespace mlir::test
