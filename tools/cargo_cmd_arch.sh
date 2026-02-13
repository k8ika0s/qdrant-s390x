case $TARGETARCH in
  "amd64")
    echo "./mold/bin/mold -run cargo"
    ;;
  "arm64" | "s390x")
    echo "cargo"
    ;;
  *)
    echo "cargo"
    ;;
esac
