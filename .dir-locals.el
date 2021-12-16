;;; Directory Local Variables
;;; For more information see (info "(emacs) Directory Variables")

;;; See https://stackoverflow.com/a/30964293/106205 and comment below (clang is picked before gcc if installed on the system, so we set both)

((c++-mode
  (flycheck-clang-language-standard . c++17)
  (flycheck-gcc-language-standard . c++17)))

