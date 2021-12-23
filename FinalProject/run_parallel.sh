for i in {1..500}; do \
    mpirun --allow-run-as-root -n 1 python3 PRRT.py $i; \
done