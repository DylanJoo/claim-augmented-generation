full_path=$1
filename=${full_path##*/}
runname=${filename/.jsonl/}

echo "Filename: $filename"
echo "Runname: $runname"

curl -X POST http://10.162.95.158:8000/submit \
  -F "run_name=$runname" \
  -F "user_id=jju8" \
  -F "description=claims retrieval pipeline testing." \
  -F "repo=https://gitlab.hltcoe.jhu.edu/scale25/decontextualize" \
  -F "git_hash=d4ba7788a17f24009bfc95981e2c1d1da5f685a1" \
  -F "run_file=@$full_path" \
