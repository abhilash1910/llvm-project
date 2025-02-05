; RUN: llvm-as %s -o %t.o
; RUN: wasm-ld %t.o -o %t.wasm --allow-undefined
; RUN: obj2yaml %t.wasm | FileCheck %s

target datalayout = "e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-n32:64-S128-ni:1:10:20"
target triple = "wasm32-unknown-unknown"

declare i32 @bar()

; Symbols such as foo which are only called indirectly are handled slightly
; differently with respect to signature checking.
declare i32 @foo()

@ptr = global ptr @foo, align 8
; Ensure access to ptr is not inlined below, even under LTO
@llvm.used = appending global [1 x ptr] [ptr @ptr], section "llvm.metadata"

define void @_start() {
  call i32 @bar()

  %addr = load ptr, ptr @ptr, align 8
  call i32 %addr()

  ret void
}

; CHECK:       - Type:            IMPORT
; CHECK-NEXT:    Imports:         
; CHECK-NEXT:      - Module:          env
; CHECK-NEXT:        Field:           bar
; CHECK-NEXT:        Kind:            FUNCTION
; CHECK-NEXT:        SigIndex:        0
; CHECK-NEXT:      - Module:          env
; CHECK-NEXT:        Field:           foo
; CHECK-NEXT:        Kind:            FUNCTION
; CHECK-NEXT:        SigIndex:        0
