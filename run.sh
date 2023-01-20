#!/bin/bash +x
#SBATCH --job-name=osc-bench
#SBATCH --partition=batch
#SBATCH --nodes=2
#SBATCH --tasks-per-node=128
#SBATCH --mem=245G              # Exclusive nodes: use max allocatable memory (*)
#SBATCH --time=4:00:00       # Run time (d-hh:mm:ss)


function run_osc {
	MPI_HOME=$1
	COMM=$2		# 0 = send/recv, 1 = PSCW, 2 = fence
	MEM=$3 		# 0 = datatype, 1 = contiguous 
	ALLOC=$4	# 0 = usr allocation, 1 = MPI_Win_allocate
	RMA=$5		# 0 = put, 1 = get
	NS=$6
	N=$7

	rm osc_* 2> /dev/null

	MPI_CC="${MPI_HOME}/bin/mpicxx -O3"
	MPI_EX="${MPI_HOME}/bin/mpiexec -n ${SLURM_NTASKS} -bind-to core -ppn ${SLURM_NTASKS_PER_NODE}"
	OPTS="-DMEM=${MEM} -DALLOC=${ALLOC} -DCOMM_MODE=${COMM} -DRMA=${RMA} -DNS=${NS} -DN=${N}"
	EXE_NAME="osc_mem${MEM}_alloc${ALLOC}_comm${COMM}_rma${RMA}"

	${MPI_CC} ${OPTS} osc.cpp -o ${EXE_NAME}
	${MPI_EX} ./${EXE_NAME}
}

#---------------------- CONSTANTS
MEM=1 # contig
RMA=0 #put
#---------------------
for version in mpich-ofi mpich-ucx
do
	if [[ "${version}" == "mpich-ofi" ]]
	then
		MPI_HOME=${HOME}/lib-MPICH-4.1rc2-OFI-1.17.0
	fi
	if [[ "${version}" == "mpich-ucx" ]]
	then
		MPI_HOME=${HOME}/lib-MPICH-4.1rc2-UCX-1.13.1
	fi

	echo "================================ ${version} =================================="

	# allocation
	for alloc in usr window
	do
		
		# number of msgs
		msg_list=( 1 2 4 8 )
		for msg in "${msg_list[@]}"
		do
			# ns -> controls size of the msg
			ns_list=( 1 2 4 6 8 16 32 )
			for ns in "${ns_list[@]}"
			do
				if [[ "${alloc}" == "usr" ]]
				then
					ALLOC=0
					#echo "------- USR ALLOC - $((${msg}*${msg}*${msg})) MSG - $((${ns}*${ns}*${ns}*4))B ------"
				fi
				if [[ "${alloc}" == "window" ]]
				then
					ALLOC=1
					#echo "------- WIN ALLOC - $((${msg}*${msg}*${msg})) MSG - $((${ns}*${ns}*${ns}*4))B ------"
				fi
				NS=${ns}
				N=$(( ${ns}* ${msg} ))


				run_osc ${MPI_HOME} 0 ${MEM} ${ALLOC} ${RMA} ${NS} ${N}
				run_osc ${MPI_HOME} 1 ${MEM} ${ALLOC} ${RMA} ${NS} ${N}
				#run_osc ${MPI_HOME} 2 ${MEM} ${ALLOC} ${RMA} ${NS} ${N}
				run_osc ${MPI_HOME} 3 ${MEM} ${ALLOC} ${RMA} ${NS} ${N}
			done
			echo " "
		done
	done

done

