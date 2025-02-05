; NOTE: Assertions have been autogenerated by utils/update_test_checks.py UTC_ARGS: --version 2
; RUN: opt -passes=slp-vectorizer -mtriple=x86_64-unknown-linux-gnu -mcpu=skx -S < %s | FileCheck %s
; RUN: opt -passes=slp-vectorizer -mtriple=x86_64-unknown-linux-gnu -mcpu=skx -S < %s -slp-threshold=-100 | FileCheck %s --check-prefix=CHECK-SLP-THRESHOLD

define void @test() {
; CHECK-LABEL: define void @test
; CHECK-SAME: () #[[ATTR0:[0-9]+]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[COND_IN_V:%.*]] = select i1 false, ptr null, ptr null
; CHECK-NEXT:    br label [[BB:%.*]]
; CHECK:       bb:
; CHECK-NEXT:    [[V:%.*]] = load i64, ptr [[COND_IN_V]], align 8
; CHECK-NEXT:    [[BV:%.*]] = icmp eq i64 [[V]], 0
; CHECK-NEXT:    [[IN_1:%.*]] = getelementptr i64, ptr [[COND_IN_V]], i64 4
; CHECK-NEXT:    [[V_1:%.*]] = load i64, ptr [[IN_1]], align 8
; CHECK-NEXT:    [[BV_1:%.*]] = icmp eq i64 [[V_1]], 0
; CHECK-NEXT:    [[IN_2:%.*]] = getelementptr i64, ptr [[COND_IN_V]], i64 8
; CHECK-NEXT:    [[V_2:%.*]] = load i64, ptr [[IN_2]], align 8
; CHECK-NEXT:    [[BV_2:%.*]] = icmp eq i64 [[V_2]], 0
; CHECK-NEXT:    [[IN_3:%.*]] = getelementptr i64, ptr [[COND_IN_V]], i64 12
; CHECK-NEXT:    [[V_3:%.*]] = load i64, ptr [[IN_3]], align 8
; CHECK-NEXT:    [[BV_3:%.*]] = icmp eq i64 [[V_3]], 0
; CHECK-NEXT:    ret void
;
; CHECK-SLP-THRESHOLD-LABEL: define void @test
; CHECK-SLP-THRESHOLD-SAME: () #[[ATTR0:[0-9]+]] {
; CHECK-SLP-THRESHOLD-NEXT:  entry:
; CHECK-SLP-THRESHOLD-NEXT:    [[COND_IN_V:%.*]] = select i1 false, ptr null, ptr null
; CHECK-SLP-THRESHOLD-NEXT:    br label [[BB:%.*]]
; CHECK-SLP-THRESHOLD:       bb:
; CHECK-SLP-THRESHOLD-NEXT:    [[TMP0:%.*]] = insertelement <4 x ptr> poison, ptr [[COND_IN_V]], i32 0
; CHECK-SLP-THRESHOLD-NEXT:    [[TMP1:%.*]] = shufflevector <4 x ptr> [[TMP0]], <4 x ptr> poison, <4 x i32> zeroinitializer
; CHECK-SLP-THRESHOLD-NEXT:    [[TMP2:%.*]] = getelementptr i64, <4 x ptr> [[TMP1]], <4 x i64> <i64 12, i64 8, i64 4, i64 0>
; CHECK-SLP-THRESHOLD-NEXT:    [[TMP3:%.*]] = call <4 x i64> @llvm.masked.gather.v4i64.v4p0(<4 x ptr> [[TMP2]], i32 8, <4 x i1> splat (i1 true), <4 x i64> poison)
; CHECK-SLP-THRESHOLD-NEXT:    [[TMP4:%.*]] = icmp eq <4 x i64> [[TMP3]], zeroinitializer
; CHECK-SLP-THRESHOLD-NEXT:    ret void
;
entry:
  %cond.in.v = select i1 false, ptr null, ptr null
  br label %bb

bb:                            ; preds = %entry
  %v = load i64, ptr %cond.in.v, align 8
  %bv = icmp eq i64 %v, 0
  %in.1 = getelementptr i64, ptr %cond.in.v, i64 4
  %v.1 = load i64, ptr %in.1, align 8
  %bv.1 = icmp eq i64 %v.1, 0
  %in.2 = getelementptr i64, ptr %cond.in.v, i64 8
  %v.2 = load i64, ptr %in.2, align 8
  %bv.2 = icmp eq i64 %v.2, 0
  %in.3 = getelementptr i64, ptr %cond.in.v, i64 12
  %v.3 = load i64, ptr %in.3, align 8
  %bv.3 = icmp eq i64 %v.3, 0
  ret void
}
