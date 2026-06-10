#!/bin/bash
# 批量上传 stimuli/ 音频到 Cloudflare R2
# 用法: bash survey/upload_to_r2.sh <ACCOUNT_ID> <R2_API_TOKEN>
#
# R2 API Token 获取方式：
# Cloudflare Dashboard → My Profile → API Tokens → Create Token
# → 选 "R2 Object Storage:Edit" 权限

ACCOUNT_ID="${1:?请提供 Account ID}"
TOKEN="${2:?请提供 R2 API Token}"
BUCKET="tts-audio"
STIMULI_DIR="$(dirname "$0")/../human_eval/stimuli"

echo "上传 $STIMULI_DIR 中的文件到 R2 存储桶 $BUCKET ..."
count=0

for f in "$STIMULI_DIR"/*; do
  fname=$(basename "$f")
  ext="${fname##*.}"
  case "$ext" in
    wav) ctype="audio/wav" ;;
    mp3) ctype="audio/mpeg" ;;
    *)   ctype="application/octet-stream" ;;
  esac

  curl -s -X PUT \
    "https://api.cloudflare.com/client/v4/accounts/${ACCOUNT_ID}/r2/buckets/${BUCKET}/objects/${fname}" \
    -H "Authorization: Bearer ${TOKEN}" \
    -H "Content-Type: ${ctype}" \
    --data-binary "@$f" > /dev/null

  echo "  ✓ $fname"
  count=$((count + 1))
done

echo ""
echo "完成！共上传 $count 个文件"
echo "公开地址: https://pub-8337bd7f96fd452dba83af102da6a1d8.r2.dev/<文件名>"
