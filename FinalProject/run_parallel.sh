for i in {1..100}; do \
    mpirun --allow-run-as-root -n 2 python3 PRRT.py $i; \
done