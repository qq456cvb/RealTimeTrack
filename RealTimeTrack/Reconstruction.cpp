//////////////////////////////////////////////////////////////////////////
// Author		:	Ngo Tien Dat
// Email		:	dat.ngo@epfl.ch
// Organization	:	EPFL
// Purpose		:	3D mesh unconstrained reconstruction
// Date			:	15 March 2012
//////////////////////////////////////////////////////////////////////////

#include "Reconstruction.h"
#include "Timer.hpp"
#include "ObjectiveFunction.h"
#include "EqualConstrFunction.h"
#include "IneqConstrFunction.h"
#include "TraceManager.hpp"

using namespace arma;

const double Reconstruction::ROBUST_SCALE = 2;

void Reconstruction::buildCorrespondenceMatrix2D( const mat& matches )
{
    int nMatches  = matches.n_rows;
    int nVertices = this->refMesh.GetNVertices();
    
    this->Binit = zeros(2*nMatches, 2*nVertices);
    this->Uinit = zeros(2*nMatches, 1);
    
//    mat allPoints = TraceManager::imageDatabase->getReferenceMesh().GetVertexCoords();
//    mat ctrlPoints = allPoints.rows(TraceManager::imageDatabase->getReferenceMesh().GetCtrlPointIDs());
//    ctrlPoints.reshape(75, 1);
//    mat allPoints2 = refMesh.GetBigParamMat() * ctrlPoints;
//    allPoints2.reshape(allPoints.n_elem/3, 3);
//    cout << "diff : " << mean(abs(allPoints - allPoints2)) << endl;
//    cout << TraceManager::imageDatabase->getRefCamera().ProjectPoints(allPoints2) << endl;
    for (int i = 0; i < nMatches; i++)
    {
        const rowvec& vid = matches(i, span(0,2));		// Vertex ids in reference image
        const rowvec& bcs = matches(i, span(3,5));		// Barycentric coordinates in reference image
        const rowvec& uvs = matches(i, span(6,7));		// Image coordinates in input image
        
//        cout << vid << endl;
//        rowvec point3D = bcs(0) * allPoints.row(vid(0)) + bcs(1) * allPoints.row(vid(1)) + bcs(2) * allPoints.row(vid(2));
//        auto point2D = TraceManager::imageDatabase->getRefCamera().ProjectAPoint(point3D.t());
//        
//        vec pt1 = TraceManager::imageDatabase->getRefCamera().ProjectAPoint(allPoints.row(vid(0)).t());
//        vec pt2 = TraceManager::imageDatabase->getRefCamera().ProjectAPoint(allPoints.row(vid(1)).t());
//        vec pt3 = TraceManager::imageDatabase->getRefCamera().ProjectAPoint(allPoints.row(vid(2)).t());
//        auto point2D0 = bcs(0) * pt1 + bcs(1) * pt2 + bcs(2) * pt3;
//        cout << point2D.t() << endl;
//        cout << point2D0.t() << endl;
        
        for (int k = 0; k <= 1; k++) {
            Binit(2*i+k, vid(0) + k*nVertices) = bcs(0);
            Binit(2*i+k, vid(1) + k*nVertices) = bcs(1);
            Binit(2*i+k, vid(2) + k*nVertices) = bcs(2);
            
//            Binit(2*i+1, vid(0) + k*nVertices) = bcs(0);
//            Binit(2*i+1, vid(1) + k*nVertices) = bcs(1);
//            Binit(2*i+1, vid(2) + k*nVertices) = bcs(2);
            
        }
        
//        cout << point2D << endl;
//        cout << uvs << endl;
        Uinit(2*i) = uvs(0);
        Uinit(2*i + 1) = uvs(1);
    }
//    mat pt2d = TraceManager::imageDatabase->getRefCamera().ProjectPoints(allPoints2);
//    pt2d.reshape(pt2d.n_elem, 1);
//    cout << pt2d << endl;
//    cout << mean(abs(Binit * pt2d - Uinit)) << endl;
//    cout << Binit << endl;
}

void Reconstruction::buildCorrespondenceMatrix( const mat& matches )
{
    int nMatches	= matches.n_rows;
    int nVertices	= this->refMesh.GetNVertices();
    
    this->Minit = zeros(2*nMatches, 3*nVertices);
    const mat& A = this->camCamera.GetA();
    
    for (int i = 0; i < nMatches; i++)
    {
        const rowvec& vid = matches(i, span(0,2));		// Vertex ids in reference image
        const rowvec& bcs = matches(i, span(3,5));		// Barycentric coordinates in reference image
        const rowvec& uvs = matches(i, span(6,7));		// Image coordinates in input image
        
        // Vertex coordinates are ordered to be [x1,...,xN, y1,...,yN, z1,...,zN]
        for (int k = 0; k <= 2; k++)
        {
            // First row
            Minit(2*i, vid(0) + k*nVertices) = bcs(0) * ( A(0,k) - uvs(0) * A(2,k) );
            Minit(2*i, vid(1) + k*nVertices) = bcs(1) * ( A(0,k) - uvs(0) * A(2,k) );
            Minit(2*i, vid(2) + k*nVertices) = bcs(2) * ( A(0,k) - uvs(0) * A(2,k) );
            
            // Second row
            Minit(2*i+1, vid(0) + k*nVertices) = bcs(0) * ( A(1,k) - uvs(1) * A(2,k) );
            Minit(2*i+1, vid(1) + k*nVertices) = bcs(1) * ( A(1,k) - uvs(1) * A(2,k) );
            Minit(2*i+1, vid(2) + k*nVertices) = bcs(2) * ( A(1,k) - uvs(1) * A(2,k) );
        }
    }
}

vec Reconstruction::computeReprojectionErrors2D( const TriangleMesh& trigMesh, const mat& matchesInit, const uvec& currentMatchIdxs, bool init)
{
//    const mat& vertexCoords = trigMesh.GetVertexCoords();
    int		   nMatches		= (int)currentMatchIdxs.n_rows;
    
    vec errors(nMatches);		// Errors of all matches
    
    // BPc - U, the error
    arma::vec BPc_U;
    if (init) {
        BPc_U = BPinit * c - Uinit;
    } else {
        BPc_U = BP * c - U;
    }
//    cout << "BPcU mean: " << arma::mean(arma::abs(BPc_U)) << endl;
    assert(BPc_U.n_rows == 2 * nMatches);
    for (int i = 0; i < nMatches; i++)
    {
        vec diff;
        diff << BPc_U(2*i) << BPc_U(2*i+1);

        errors(i) = norm(diff, 2);
//        cout << diff << endl;
//        cout << errors(i) << endl;
    }
    
    return errors;
}

vec Reconstruction::computeReprojectionErrors( const TriangleMesh& trigMesh, const mat& matchesInit, const uvec& currentMatchIdxs )
{
    const mat& vertexCoords = trigMesh.GetVertexCoords();
    int		   nMatches		= currentMatchIdxs.n_rows;
    
    vec errors(nMatches);		// Errors of all matches
    
    for (int i = 0; i < nMatches; i++)
    {
        // Facet (3 vertex IDs) that contains the matching point
        int idx  = currentMatchIdxs(i);
        int vId1 = (int)matchesInit(idx, 0);
        int vId2 = (int)matchesInit(idx, 1);
        int vId3 = (int)matchesInit(idx, 2);
        
        // 3D vertex coordinates
        const rowvec& vertex1Coords = vertexCoords.row(vId1);
        const rowvec& vertex2Coords = vertexCoords.row(vId2);
        const rowvec& vertex3Coords = vertexCoords.row(vId3);
        
        double bary1 = matchesInit(idx, 3);
        double bary2 = matchesInit(idx, 4);
        double bary3 = matchesInit(idx, 5);
        
        // 3D feature point
        rowvec point3D = bary1*vertex1Coords + bary2*vertex2Coords + bary3*vertex3Coords;
        
        // TODO: Implement this function in stead of call projecting function for a single point. This can save expense of function calls
        // Projection
        vec point2D = camCamera.ProjectAPoint(point3D.t());
        
        vec matchingPoint(2);
        matchingPoint(0) = matchesInit(idx, 6);
        matchingPoint(1) = matchesInit(idx, 7);
        
//        cout << point2D - matchingPoint << endl;
        errors(i) = norm(point2D - matchingPoint, 2);
//        cout << errors(i) << endl;
    }
    
    return errors;
}

void Reconstruction::reconstructPlanarUnconstr2D( const arma::uvec& matchIdxs, double wr, LaplacianMesh& resMesh )
{
    Timer timer;
    
//    const mat&	paramMat = this->refMesh.GetParamMatrix();	// Parameterization matrix
    
    // Build the matrix MPwAP = [MP; wr*AP] and compute: (MPwAP)' * (MPwAP)
    this->computeCurrentMatrices2D( matchIdxs, wr);
    
    // --------------- Eigen value decomposition --------------------------
    //cout << "Eigen(): " << timer.getElapsedTimeInMilliSec() << " ms"<< endl;
//    c = solve(BP, U);
    mat BPtBPwwAPtAP = BP.t() * BP + wr * wr * this->APtAP2D;
    mat BPtU = BP.t() * U;
    c = solve (BPtBPwwAPtAP, BPtU);
}

void Reconstruction::updateMesh( LaplacianMesh& resMesh )
{
    const mat&	paramMat = this->refMesh.GetParamMatrix();
    
    mat matC = TraceManager::imageDatabase->getRefCamera().ReprojectPoints(reshape(c, refMesh.GetNCtrlPoints(), 2));
//    mat matC = reshape(c, refMesh.GetNCtrlPoints(), 3);

    // Update vertex coordinates
    resMesh.SetVertexCoords(paramMat * matC);

    // Resulting mesh yields a correct projection on image but it does not preserve lengths.
    // So we need to compute the scale factor and multiply with matC

    // Determine on which side the mesh lies: -Z or Z
    double meanZ = mean(resMesh.GetVertexCoords().col(2));
    int globalSign = meanZ > 0 ? 1 : -1;

    const vec& resMeshEdgeLens = resMesh.ComputeEdgeLengths();
    const vec& refMeshEdgeLens = refMesh.GetEdgeLengths();

    // this seems no difference if do not scale the points?
    double scale = globalSign * norm(refMeshEdgeLens, 2) / norm(resMeshEdgeLens, 2);

    // Update vertex coordinates
    resMesh.SetVertexCoords(scale * paramMat * matC);
}

void Reconstruction::reconstructPlanarUnconstr( const uvec& matchIdxs, double wr, LaplacianMesh& resMesh )
{
    Timer timer;
    
    const mat&	paramMat = this->refMesh.GetParamMatrix();	// Parameterization matrix
    
    // Build the matrix MPwAP = [MP; wr*AP] and compute: (MPwAP)' * (MPwAP)
    this->computeCurrentMatrices( matchIdxs, wr);
    
    // --------------- Eigen value decomposition --------------------------
    mat V;
    vec s;
    timer.start();
    
    // 此处最小的特征值只是其中一个解，在下面会对这个解进行Scale使得它的边长和ref mesh的边长相等
    eig_sym(s, V, this->MPwAPtMPwAP);
    timer.stop();
//    cout << "Eigen(): " << timer.getElapsedTimeInMilliSec() << " ms"<< endl;
    const vec& c = V.col(0);
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    mat matC = reshape(c, refMesh.GetNCtrlPoints(), 3);
    
    // Update vertex coordinates
    resMesh.SetVertexCoords(paramMat * matC);
    
    // Resulting mesh yields a correct projection on image but it does not preserve lengths.
    // So we need to compute the scale factor and multiply with matC
    
    // Determine on which side the mesh lies: -Z or Z
    double meanZ = mean(resMesh.GetVertexCoords().col(2));
    int globalSign = meanZ > 0 ? 1 : -1;
    
    const vec& resMeshEdgeLens = resMesh.ComputeEdgeLengths();
    const vec& refMeshEdgeLens = refMesh.GetEdgeLengths();
    
    double scale = globalSign * norm(refMeshEdgeLens, 2) / norm(resMeshEdgeLens, 2);
    
    // Update vertex coordinates
    resMesh.SetVertexCoords(scale * paramMat * matC);
}

void Reconstruction::ReconstructPlanarUnconstrOnce( const mat& matchesInit, LaplacianMesh& resMesh)
{
    // Input check
    if (matchesInit.n_rows == 0) {
        return;
    }
    
    Timer timer;
    
    double meanZ = mean(resMesh.GetVertexCoords().col(2));
    
    // scale wr due to distance : 0 - 200
    double	wr		= this->wrInit  * (meanZ / 200.);			// Currently used regularization weight
//    double	radius	= this->radiusInit;		// Currently used radius of the estimator
//    vec		reprojErrors;					// Reprojection errors
    
    // First, we need to build the correspondent matrix with all given matches to avoid re-computation
    this->buildCorrespondenceMatrix(matchesInit);
    
    // Then compute MPinit. Function reconstructPlanarUnconstr() will use part of MPinit w.r.t currently used matches
    this->MPinit = this->Minit * this->refMesh.GetBigParamMat();
    
    uvec matchesInitIdxs = linspace<uvec>(0, matchesInit.n_rows-1, matchesInit.n_rows);
    
    // Currently used matches represented by their indices. Initially, use all matches: [0,1,2..n-1]
    
    this->reconstructPlanarUnconstr(matchesInitIdxs, wr, resMesh);
}

void Reconstruction::ReconstructPlanarUnconstrIter2D( const mat& matchesInit, LaplacianMesh& resMesh, uvec& inlierMatchIdxs )
{
    // Input check
    if (matchesInit.n_rows == 0) {
        inlierMatchIdxs.resize(0);
        return;
    }
    
    Timer timer;
    
    double	wr		= this->wrInit;			// Currently used regularization weight
    double	radius	= this->radiusInit * 4;		// Currently used radius of the estimator
    vec		reprojErrors;					// Reprojection errors
    
    // First, we need to build the correspondent matrix with all given matches to avoid re-computation
    this->buildCorrespondenceMatrix2D(matchesInit);
    
    // Then compute MPinit. Function reconstructPlanarUnconstr() will use part of MPinit w.r.t currently used matches
    this->BPinit = this->Binit * this->refMesh.GetBigParamMat2D();
    
    uvec matchesInitIdxs = linspace<uvec>(0, matchesInit.n_rows-1, matchesInit.n_rows);
    
    // Currently used matches represented by their indices. Initially, use all matches: [0,1,2..n-1]
    inlierMatchIdxs = matchesInitIdxs;
    
    for (int i = 0; i < nUncstrIters; i++)
    {
        this->reconstructPlanarUnconstr2D(inlierMatchIdxs, wr, resMesh);
        
        // If it is the final iteration, break and don't update "inlierMatchIdxs" or "weights", "radius"
        if (i == nUncstrIters - 1) {
//            updateMesh(resMesh);
//            computeCurrentBPwAP(inlierMatchIdxs, wr);
            //cout << "Current radius: " << radius << endl;
            //cout << "Current wr: " << wr << endl;
            //Reconstruction::computeCurrentMatrices( currentMatchIdxs, 325 );	// For Fern
            break;
        }
        
        // Otherwise, remove outliers
        int iterTO = nUncstrIters - 2;
        if (i >= iterTO)
            reprojErrors = this->computeReprojectionErrors2D(resMesh, matchesInit, matchesInitIdxs, true);
        else
            reprojErrors = this->computeReprojectionErrors2D(resMesh, matchesInit, inlierMatchIdxs, false);
        
        uvec idxs = find( reprojErrors < radius );
//        cout << reprojErrors << endl;
        if ( idxs.n_elem == 0 )
            break;
        
        if (i >= iterTO)
            inlierMatchIdxs = matchesInitIdxs.elem( idxs );
        else
            inlierMatchIdxs = inlierMatchIdxs.elem( idxs );
        
        // Update parameters
        wr = wr / Reconstruction::ROBUST_SCALE;
        radius	= radius / Reconstruction::ROBUST_SCALE;
    }
}

void Reconstruction::ReconstructPlanarUnconstrIter( const mat& matchesInit, LaplacianMesh& resMesh, uvec& inlierMatchIdxs )
{
    // Input check
    if (matchesInit.n_rows == 0) {
        inlierMatchIdxs.resize(0);
        return;
    }
    
    Timer timer;
    
    double	wr		= this->wrInit;			// Currently used regularization weight
    double	radius	= this->radiusInit;		// Currently used radius of the estimator
    vec		reprojErrors;					// Reprojection errors
    
    // First, we need to build the correspondent matrix with all given matches to avoid re-computation
    this->buildCorrespondenceMatrix(matchesInit);
    
    // Then compute MPinit. Function reconstructPlanarUnconstr() will use part of MPinit w.r.t currently used matches
    this->MPinit = this->Minit * this->refMesh.GetBigParamMat();
    
    uvec matchesInitIdxs = linspace<uvec>(0, matchesInit.n_rows-1, matchesInit.n_rows);
    
    // Currently used matches represented by their indices. Initially, use all matches: [0,1,2..n-1]
    inlierMatchIdxs = matchesInitIdxs;
    
    for (int i = 0; i < nUncstrIters; i++)
    {
        
        
        // If it is the final iteration, break and don't update "inlierMatchIdxs" or "weights", "radius"
        if (i == nUncstrIters - 1) {
            double meanZ = mean(resMesh.GetVertexCoords().col(2));
            
            this->reconstructPlanarUnconstr(inlierMatchIdxs, this->wrInit * (meanZ / 200.), resMesh);
            //cout << "Current radius: " << radius << endl;
            //cout << "Current wr: " << wr << endl;
            //Reconstruction::computeCurrentMatrices( currentMatchIdxs, 325 );	// For Fern
            break;
        } else {
            this->reconstructPlanarUnconstr(inlierMatchIdxs, wr, resMesh);
        }
        
        // Otherwise, remove outliers
        int iterTO = nUncstrIters - 2;
        if (i >= iterTO)
            reprojErrors = this->computeReprojectionErrors(resMesh, matchesInit, matchesInitIdxs);
        else
            reprojErrors = this->computeReprojectionErrors(resMesh, matchesInit, inlierMatchIdxs);
        
        uvec idxs = find( reprojErrors < radius );
        if ( idxs.n_elem == 0 )
            break;
        
        if (i >= iterTO)
            inlierMatchIdxs = matchesInitIdxs.elem( idxs );
        else
            inlierMatchIdxs = inlierMatchIdxs.elem( idxs );
        
        // Update parameters
        wr		= wr 	 / Reconstruction::ROBUST_SCALE;
        radius	= radius / Reconstruction::ROBUST_SCALE;
    }
}

void Reconstruction::computeCurrentMatrices2D(const arma::uvec &matchIdxs, double wr)
{
    int	nMatches = matchIdxs.n_rows;			// Number of currently used matches
    
    // Build matrix currentMP by taking some rows of MP corresponding to currently used match indices
    Timer timer;
    timer.start();
    mat currentBP(2 * nMatches, this->BPinit.n_cols);
    mat currentU(2 * nMatches, this->Uinit.n_cols);
    for (int i = 0; i < nMatches; i++)
    {
        currentBP.rows(2*i, 2*i+1) = this->BPinit.rows(2*matchIdxs(i), 2*matchIdxs(i) + 1);
        currentU.rows(2*i, 2*i+1) = this->Uinit.rows(2*matchIdxs(i), 2*matchIdxs(i) + 1);
        
    }
    timer.stop();
    BP = currentBP;
    U = currentU;
    //cout << "Build current BP matrix: " << timer.getElapsedTimeInMilliSec() << " ms"<< endl;
    
    //cout << "solve (BPtBP+wI)c = BPtU: " << timer.getElapsedTimeInMilliSec() << " ms"<< endl;
}

// this is in 3D space not 2D!!
void Reconstruction::computeCurrentBPwAP(const arma::uvec &matchIdxs, double wr)
{
    int	nMatches = matchIdxs.n_rows;			// Number of currently used matches
    
    // Build matrix currentMP by taking some rows of MP corresponding to currently used match indices
    Timer timer;
    timer.start();
    mat currentB(nMatches, this->Binit.n_cols / 2);
    for (int i = 0; i < nMatches; i++)
    {
        currentB.row(i) = this->Binit(2*matchIdxs(i), arma::span(0, Binit.n_cols / 2 - 1));
    }
    timer.stop();
    
    mat currentBP = kron(eye(3, 3), currentB) * refMesh.GetBigParamMat();
    BPwAP = join_cols( currentBP, wr*refMesh.GetBigAP() );
}

void Reconstruction::computeCurrentMatrices( const uvec& matchIdxs, double wr )
{
    int	nMatches = matchIdxs.n_rows;			// Number of currently used matches
    
    // Build matrix currentMP by taking some rows of MP corresponding to currently used match indices
    Timer timer;
    timer.start();
    mat currentMP(2 * nMatches, this->MPinit.n_cols);
    for (int i = 0; i < nMatches; i++)
    {
        currentMP.rows(2*i, 2*i+1) = this->MPinit.rows(2*matchIdxs(i), 2*matchIdxs(i) + 1);
    }
    timer.stop();
    //cout << "Build current MP matrix: " << timer.getElapsedTimeInMilliSec() << " ms"<< endl;
    
    timer.start();
    MPwAP 		= join_cols( currentMP, wr*refMesh.GetBigAP() );	// TODO: Avoid computing this. Only needed in the last iteration
    MPwAPtMPwAP = currentMP.t() * currentMP + wr*wr * this->APtAP;
    timer.stop();
    //cout << "Build (MPwAP)' * (MPwAP): " << timer.getElapsedTimeInMilliSec() << " ms"<< endl;
}

void Reconstruction::ReconstructEqualConstr( const vec& cInit, LaplacianMesh& resMesh )
{
    const mat& paramMat = refMesh.GetParamMatrix();
    const mat& bigP		= refMesh.GetBigParamMat();
    
    if (cOptimal.is_empty()) {
        cOptimal = reshape(refMesh.GetVertexCoords().rows(refMesh.GetCtrlPointIDs()), refMesh.GetNCtrlPoints()*3, 1 );
    }
    
    // Objective function: use MPwAP which was already computed in unconstrained reconstruction
    ObjectiveFunction *objtFunction;
    
    if ( this->useTemporal && !isFirstFrame ) {
        objtFunction = new ObjectiveFunction( this->GetMPwAP(), this->timeSmoothAlpha, cOptimal );
    } else {
        objtFunction = new ObjectiveFunction( this->GetMPwAP() );
    }
    
    // Constrained function
    EqualConstrFunction cstrFunction( bigP, refMesh.GetEdges(), refMesh.GetEdgeLengths() );
    
    if ( this->usePrevFrameToInit && !isFirstFrame )
        cOptimal = equalConstrOptimize.OptimizeNullSpace(cOptimal, *objtFunction, cstrFunction);
    else
        cOptimal = equalConstrOptimize.OptimizeNullSpace(cInit, *objtFunction, cstrFunction);
    
    mat cOptimalMat = reshape(cOptimal, refMesh.GetNCtrlPoints(), 3);
    if ( cOptimalMat(0,2) < 0 ) {		// Change the sign if the reconstruction is behind the camera. This happens because we take cOptimal as initial value for constrained optimization.
        cOptimalMat = -cOptimalMat;
    }
    
    // Update vertex coordinates
    resMesh.SetVertexCoords(paramMat*cOptimalMat);
    
    isFirstFrame = false;
    delete objtFunction;
}

void Reconstruction::ReconstructIneqConstr( const vec& cInit, LaplacianMesh& resMesh )
{
    const mat& paramMat = refMesh.GetParamMatrix();
    const mat& bigP		= refMesh.GetBigParamMat();
    
    if (cOptimal.is_empty()) {
        cOptimal = reshape(refMesh.GetVertexCoords().rows(refMesh.GetCtrlPointIDs()), refMesh.GetNCtrlPoints()*3, 1 );
    }
    
    // Objective function: use MPwAP which was already computed in unconstrained reconstruction
    ObjectiveFunction *objtFunction;
    
    if ( this->useTemporal && !isFirstFrame  ) {
        objtFunction = new ObjectiveFunction( this->GetMPwAP(), this->timeSmoothAlpha, cOptimal );
    } else {
        objtFunction = new ObjectiveFunction( this->GetMPwAP() );
    }
    
    // Constrained function
    IneqConstrFunction cstrFunction( bigP, refMesh.GetEdges(), refMesh.GetEdgeLengths() );
    
    if ( this->usePrevFrameToInit && !isFirstFrame )
        cOptimal = ineqConstrOptimize.OptimizeLagrange(cOptimal, *objtFunction, cstrFunction);
    else
        cOptimal = ineqConstrOptimize.OptimizeLagrange(cInit, *objtFunction, cstrFunction);
    
    mat cOptimalMat = reshape(cOptimal, refMesh.GetNCtrlPoints(), 3);
    if ( cOptimalMat(0,2) < 0 ) {		// Change the sign if the reconstruction is behind the camera. This happens because we take cOptimal as initial value for constrained optimization.
        cOptimalMat = -cOptimalMat;
    }
    
    // Update vertex coordinates
    resMesh.SetVertexCoords(paramMat*cOptimalMat);
    
    isFirstFrame = false;
    delete objtFunction;
}

void Reconstruction::ReconstructSoftConstr(const arma::vec& cInit, LaplacianMesh &resMesh) {
    const mat& paramMat = refMesh.GetParamMatrix();
    const mat& bigP		= refMesh.GetBigParamMat();
    
    if (cOptimal.is_empty()) {
        cOptimal = reshape(refMesh.GetVertexCoords().rows(refMesh.GetCtrlPointIDs()), refMesh.GetNCtrlPoints()*3, 1 );
    }
    
    SoftConstrFunction function( bigP, this->GetMPwAP(), refMesh.GetEdges(), refMesh.GetEdgeLengths());

    if ( this->usePrevFrameToInit && !isFirstFrame )
        cOptimal = softConstrOptimize.OptimizeLagrange(cOptimal, function, resMesh);
    else
        cOptimal = softConstrOptimize.OptimizeLagrange(cInit, function, resMesh);
    
    arma::mat cOptimalMat = arma::mat(reshape(cOptimal, refMesh.GetNCtrlPoints(), 3));
    if ( mean(cOptimalMat.col(2)) < 0 ) {		// Change the sign if the reconstruction is behind the camera. This happens because we take cOptimal as initial value for constrained optimization.
        cOptimalMat = -cOptimalMat;
    }
    
    // Update vertex coordinates
    resMesh.SetVertexCoords(paramMat*(cOptimalMat));
    
    isFirstFrame = false;
}

void Reconstruction::ReconstructSoftConstr2D(const arma::vec& cInit, LaplacianMesh &resMesh) {
    const mat& paramMat = refMesh.GetParamMatrix();
    const mat& bigP		= refMesh.GetBigParamMat();
    
    if (cOptimal.is_empty()) {
        cOptimal = reshape(refMesh.GetVertexCoords().rows(refMesh.GetCtrlPointIDs()), refMesh.GetNCtrlPoints()*3, 1 );
    }
    
    SoftConstrFunction2D function( bigP, this->GetBPwAP(), refMesh.GetEdges(), refMesh.GetEdgeLengths());
    
    if ( this->usePrevFrameToInit && !isFirstFrame )
        cOptimal = softConstrOptimize.OptimizeLagrange(cOptimal, function, resMesh);
    else
        cOptimal = softConstrOptimize.OptimizeLagrange(cInit, function, resMesh);
    
    arma::mat cOptimalMat = arma::mat(reshape(cOptimal, refMesh.GetNCtrlPoints(), 3));
    if ( mean(cOptimalMat.col(2)) < 0 ) {		// Change the sign if the reconstruction is behind the camera. This happens because we take cOptimal as initial value for constrained optimization.
        cOptimalMat = -cOptimalMat;
    }
    
    // Update vertex coordinates
    resMesh.SetVertexCoords(paramMat*(cOptimalMat));
    
    isFirstFrame = false;
}



