make
est_return_val=$?

if [ "$est_return_val" -ne "0" ]; then
  echo -e "\e[96mCompilation failed with error code [$est_return_val]\e[0m"
  exit 1
fi

./main "../uimg5.jpg"
