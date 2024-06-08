void setup_cu(double *data, int n, int n_batch);
void doit_cu(int iter);
void finalize_cu(double *result,int n,int n_batch);
void setup_cu_2d(double *data, int nx, int ny, int n_batch);
void doit_cu_2d(int iter);
void finalize_cu_2d(double *result,int nx,int ny,int n_batch);