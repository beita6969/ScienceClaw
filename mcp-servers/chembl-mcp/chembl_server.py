from typing import Any, List, Dict, Callable, TypeVar, Optional
import asyncio
import logging
import functools
import time
from mcp.server.fastmcp import FastMCP
import chembl_webresource_client
from chembl_webresource_client.new_client import new_client
from chembl_webresource_client.utils import utils

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize FastMCP server
mcp = FastMCP("chembl")

# Define return type variable
T = TypeVar('T')

# Async timeout decorator
def async_timeout(seconds: int):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                logging.error(f"Function {func.__name__} execution timed out (exceeded {seconds} seconds)")
                raise TimeoutError(f"Function execution exceeded {seconds} seconds")
        return wrapper
    return decorator

# Error handling decorator
def error_handler(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            start_time = time.time()
            result = await func(*args, **kwargs)
            end_time = time.time()
            logging.info(f"{func.__name__} execution time: {end_time - start_time:.2f} seconds")
            return result
        except TimeoutError as e:
            logging.error(f"{func.__name__} timeout error: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"{func.__name__} execution error: {str(e)}")
            raise
    return wrapper


@mcp.tool()
@error_handler
@async_timeout(10)
async def example_activity(assay_chembl_id: str) -> List[Dict[str, Any]]:
    """Get activity data for the specified assay_chembl_id
    
    Args:
        assay_chembl_id: ChEMBL assay ID
        
    Returns:
        List of activity data
    """
    client = new_client
    activities = client.activity.filter(assay_chembl_id=assay_chembl_id)
    return list(activities)

@mcp.tool()
@error_handler
@async_timeout(10)
async def example_activity_supplementary_data_by_activity(activity_chembl_id: str) -> List[Dict[str, Any]]:
    """Get supplementary activity data for the specified activity_chembl_id
    
    Args:
        activity_chembl_id: ChEMBL activity ID
        
    Returns:
        List of supplementary activity data
    """
    client = new_client
    activity_supp_data = client.activity_supplementary_data_by_activity.filter(activity_chembl_id=activity_chembl_id)
    return list(activity_supp_data)

@mcp.tool()
@error_handler
@async_timeout(10)
async def example_assay(assay_type: str) -> List[Dict[str, Any]]:
    """Get assay data for the specified type
    
    Args:
        assay_type: Assay type
        
    Returns:
        List of assay data
    """
    client = new_client
    assays = client.assay.filter(assay_type=assay_type)
    return list(assays)

@mcp.tool()
@error_handler
@async_timeout(10)
async def example_assay_class(assay_class_type: str) -> List[Dict[str, Any]]:
    """Get assay classification data for the specified type
    
    Args:
        assay_class_type: Assay classification type
        
    Returns:
        List of assay classification data
    """
    client = new_client
    assay_classes = client.assay_class.filter(assay_class_type=assay_class_type)
    return list(assay_classes)

@mcp.tool()
@error_handler
@async_timeout(10)
async def example_atc_class(level1: str) -> List[Dict[str, Any]]:
    """Get ATC classification data for the specified level1
    
    Args:
        level1: Level1 value of ATC classification
        
    Returns:
        List of ATC classification data
    """
    client = new_client
    atc_classes = client.atc_class.filter(level1=level1)
    return list(atc_classes)

@mcp.tool()
@error_handler
@async_timeout(10)
async def example_binding_site(site_name: str) -> List[Dict[str, Any]]:
    """Get binding site data for the specified name
    
    Args:
        site_name: Binding site name
        
    Returns:
        List of binding site data
    """
    client = new_client
    binding_sites = client.binding_site.filter(site_name=site_name)
    return list(binding_sites)

@mcp.tool()
@error_handler
@async_timeout(10)
async def example_biotherapeutic(biotherapeutic_type: str) -> List[Dict[str, Any]]:
    """Get biotherapeutic data for the specified type
    
    Args:
        biotherapeutic_type: Biotherapeutic type
        
    Returns:
        List of biotherapeutic data
    """
    client = new_client
    biotherapeutics = client.biotherapeutic.filter(biotherapeutic_type=biotherapeutic_type)
    return list(biotherapeutics)

@mcp.tool()
@error_handler
@async_timeout(10)
async def example_cell_line(cell_line_name: str) -> List[Dict[str, Any]]:
    """Get cell line data for the specified name
    
    Args:
        cell_line_name: Cell line name
        
    Returns:
        List of cell line data
    """
    client = new_client
    cell_lines = client.cell_line.filter(cell_line_name=cell_line_name)
    return list(cell_lines)

@mcp.tool()
@error_handler
@async_timeout(10)
async def example_chembl_id_lookup(available_type: str, q: str) -> List[Dict[str, Any]]:
    """Look up ChEMBL IDs for the specified type and query
    
    Args:
        available_type: Available type
        q: Query string
        
    Returns:
        List of ChEMBL IDs
    """
    client = new_client
    chembl_ids = client.chembl_id_lookup.filter(available_type=available_type, q=q)
    return list(chembl_ids)

@mcp.tool()
@error_handler
@async_timeout(10)
async def example_chembl_release() -> List[Dict[str, Any]]:
    """Get all ChEMBL release information
    
    Returns:
        List of ChEMBL release information
    """
    client = new_client
    chembl_releases = client.chembl_release.all()
    return list(chembl_releases)

@mcp.tool()
@error_handler
@async_timeout(10)
async def example_compound_record(compound_name: str) -> List[Dict[str, Any]]:
    """Get compound records for the specified name
    
    Args:
        compound_name: Compound name
        
    Returns:
        List of compound records
    """
    client = new_client
    compound_records = client.compound_record.filter(compound_name=compound_name)
    return list(compound_records)

@mcp.tool()
@error_handler
@async_timeout(10)
async def example_compound_structural_alert(alert_name: str) -> List[Dict[str, Any]]:
    """Get compound structural alerts for the specified name
    
    Args:
        alert_name: Alert name
        
    Returns:
        List of compound structural alerts
    """
    client = new_client
    structural_alerts = client.compound_structural_alert.filter(alert_name=alert_name)
    return list(structural_alerts)

@mcp.tool()
@error_handler
@async_timeout(10)
async def example_description(description_type: str) -> List[Dict[str, Any]]:
    """Get description data for the specified type
    
    Args:
        description_type: Description type
        
    Returns:
        List of description data
    """
    client = new_client
    descriptions = client.description.filter(description_type=description_type)
    return list(descriptions)

@mcp.tool()
@error_handler
@async_timeout(10)
async def example_document(journal: str) -> List[Dict[str, Any]]:
    """Get document data for the specified journal
    
    Args:
        journal: Journal name
        
    Returns:
        List of document data
    """
    client = new_client
    documents = client.document.filter(journal=journal)
    return list(documents)

@mcp.tool()
@error_handler
@async_timeout(10)
async def example_drug(drug_type: str) -> List[Dict[str, Any]]:
    """Get drug data for the specified type
    
    Args:
        drug_type: Drug type
        
    Returns:
        List of drug data
    """
    client = new_client
    drugs = client.drug.filter(drug_type=drug_type)
    return list(drugs)

@mcp.tool()
@error_handler
@async_timeout(10)
async def example_drug_indication(mesh_heading: str) -> List[Dict[str, Any]]:
    """Get drug indication data for the specified MeSH heading
    
    Args:
        mesh_heading: MeSH heading
        
    Returns:
        List of drug indication data
    """
    client = new_client
    drug_indications = client.drug_indication.filter(mesh_heading=mesh_heading)
    return list(drug_indications)

@mcp.tool()
@error_handler
@async_timeout(10)
async def example_drug_warning(meddra_term: str) -> List[Dict[str, Any]]:
    """Get drug warning data for the specified MedDRA term
    
    Args:
        meddra_term: MedDRA term
        
    Returns:
        List of drug warning data
    """
    client = new_client
    drug_warnings = client.drug_warning.filter(meddra_term=meddra_term)
    return list(drug_warnings)

@mcp.tool()
@error_handler
@async_timeout(10)
async def example_go_slim(go_slim_term: str) -> List[Dict[str, Any]]:
    """Get data for the specified GO Slim term
    
    Args:
        go_slim_term: GO Slim term
        
    Returns:
        List of GO Slim data
    """
    client = new_client
    go_slims = client.go_slim.filter(go_slim_term=go_slim_term)
    return list(go_slims)

@mcp.tool()
@error_handler
@async_timeout(10)
async def example_mechanism(mechanism_of_action: str) -> List[Dict[str, Any]]:
    """Get data for the specified mechanism of action
    
    Args:
        mechanism_of_action: Mechanism of action
        
    Returns:
        List of mechanism data
    """
    client = new_client
    mechanisms = client.mechanism.filter(mechanism_of_action=mechanism_of_action)
    return list(mechanisms)

@mcp.tool()
@error_handler
@async_timeout(10)
async def example_molecule(molecule_type: str) -> List[Dict[str, Any]]:
    """Get molecule data for the specified type
    
    Args:
        molecule_type: Molecule type
        
    Returns:
        List of molecule data
    """
    client = new_client
    molecules = client.molecule.filter(molecule_type=molecule_type)
    return list(molecules)

@mcp.tool()
@error_handler
@async_timeout(10)
async def example_molecule_form(form_description: str) -> List[Dict[str, Any]]:
    """Get molecule form data for the specified description
    
    Args:
        form_description: Form description
        
    Returns:
        List of molecule form data
    """
    client = new_client
    molecule_forms = client.molecule_form.filter(form_description=form_description)
    return list(molecule_forms)

@mcp.tool()
@error_handler
@async_timeout(10)
async def example_organism(tax_id: int) -> List[Dict[str, Any]]:
    """Get organism data for the specified taxonomy ID
    
    Args:
        tax_id: Taxonomy ID
        
    Returns:
        List of organism data
    """
    client = new_client
    organisms = client.organism.filter(tax_id=tax_id)
    return list(organisms)

@mcp.tool()
@error_handler
@async_timeout(10)
async def example_protein_classification(protein_class_name: str) -> List[Dict[str, Any]]:
    """Get protein classification data for the specified class name
    
    Args:
        protein_class_name: Protein class name
        
    Returns:
        List of protein classification data
    """
    client = new_client
    protein_classifications = client.protein_classification.filter(protein_class_name=protein_class_name)
    return list(protein_classifications)

@mcp.tool()
@error_handler
@async_timeout(10)
async def example_source(source_description: str) -> List[Dict[str, Any]]:
    """Get source information for the specified description
    
    Args:
        source_description: Source description
        
    Returns:
        List of source information
    """
    client = new_client
    sources = client.source.filter(source_description=source_description)
    return list(sources)

@mcp.tool()
@error_handler
@async_timeout(10)
async def example_target(target_type: str) -> List[Dict[str, Any]]:
    """Get target data for the specified type
    
    Args:
        target_type: Target type
        
    Returns:
        List of target data
    """
    client = new_client
    targets = client.target.filter(target_type=target_type)
    return list(targets)

@mcp.tool()
@error_handler
@async_timeout(10)
async def example_target_component(component_type: str) -> List[Dict[str, Any]]:
    """Get target component data for the specified type
    
    Args:
        component_type: Component type
        
    Returns:
        List of target component data
    """
    client = new_client
    target_components = client.target_component.filter(component_type=component_type)
    return list(target_components)

@mcp.tool()
@error_handler
@async_timeout(10)
async def example_target_relation(relationship_type: str) -> List[Dict[str, Any]]:
    """Get target relationship data for the specified relationship type
    
    Args:
        relationship_type: Relationship type
        
    Returns:
        List of target relationship data
    """
    client = new_client
    target_relations = client.target_relation.filter(relationship_type=relationship_type)
    return list(target_relations)

@mcp.tool()
@error_handler
@async_timeout(10)
async def example_tissue(tissue_name: str) -> List[Dict[str, Any]]:
    """Get tissue data for the specified name
    
    Args:
        tissue_name: Tissue name
        
    Returns:
        List of tissue data
    """
    client = new_client
    tissues = client.tissue.filter(tissue_name=tissue_name)
    return list(tissues)

@mcp.tool()
@error_handler
@async_timeout(10)
async def example_xref_source(xref_name: str) -> List[Dict[str, Any]]:
    """Get cross-reference source data for the specified name
    
    Args:
        xref_name: Cross-reference source name
        
    Returns:
        List of cross-reference source data
    """
    client = new_client
    xref_sources = client.xref_source.filter(xref_name=xref_name)
    return list(xref_sources)

@mcp.tool()
@error_handler
@async_timeout(5)
async def example_canonicalizeSmiles(smiles: str) -> str:
    """Convert SMILES string to canonical form
    
    Args:
        smiles: SMILES string
        
    Returns:
        Canonicalized SMILES string
    """
    canonical_smiles = utils.canonicalizeSmiles(smiles)
    return canonical_smiles

@mcp.tool()
@error_handler
@async_timeout(5)
async def example_chemblDescriptors(smiles: str) -> Dict[str, Any]:
    """Get ChEMBL descriptors for the SMILES string
    
    Args:
        smiles: SMILES string
        
    Returns:
        Dictionary of ChEMBL descriptors
    """
    descriptors = utils.chemblDescriptors(smiles)
    return descriptors

@mcp.tool()
@error_handler
@async_timeout(5)
async def example_description_utils(chembl_id: str) -> str:
    """
    Get description information for the ChEMBL ID
    
    Args:
        chembl_id: ChEMBL ID
        
    Returns:
        Description information
    """
    description = utils.description(chembl_id)
    return description

@mcp.tool()
@error_handler
@async_timeout(5)
async def example_descriptors(smiles: str) -> Dict[str, Any]:
    """
    Get descriptors for the SMILES string
    
    Args:
        smiles: SMILES string
        
    Returns:
        Dictionary of descriptors
    """
    descriptors = utils.descriptors(smiles)
    return descriptors

@mcp.tool()
@error_handler
@async_timeout(5)
async def example_getParent(chembl_id: str) -> str:
    """
    Get parent ChEMBL ID for the given ChEMBL ID
    
    Args:
        chembl_id: ChEMBL ID
        
    Returns:
        Parent ChEMBL ID
    """
    parent = utils.getParent(chembl_id)
    return parent

@mcp.tool()
@error_handler
@async_timeout(5)
async def example_highlightSmilesFragmentSvg(smiles: str, fragment: str) -> str:
    """
    Generate SVG image with highlighted fragment for SMILES string
    
    Args:
        smiles: SMILES string
        fragment: Fragment to highlight
        
    Returns:
        SVG image string
    """
    highlighted_svg = utils.highlightSmilesFragmentSvg(smiles, fragment)
    return highlighted_svg

@mcp.tool()
@error_handler
@async_timeout(5)
async def example_inchi2inchiKey(inchi: str) -> str:
    """
    Convert InChI to InChI Key
    
    Args:
        inchi: InChI string
        
    Returns:
        InChI Key
    """
    inchi_key = utils.inchi2inchiKey(inchi)
    return inchi_key

@mcp.tool()
@error_handler
@async_timeout(5)
async def example_inchi2svg(inchi: str) -> str:
    """
    Convert InChI to SVG image
    
    Args:
        inchi: InChI string
        
    Returns:
        SVG image string
    """
    inchi_svg = utils.inchi2svg(inchi)
    return inchi_svg
    # print("InChI SVG:", inchi_svg)  # Skipping printing SVG

@mcp.tool()
@error_handler
@async_timeout(5)
async def example_is3D(smiles: str) -> bool:
    """
    Check if SMILES string represents a 3D structure
    
    Args:
        smiles: SMILES string
        
    Returns:
        True if 3D structure, False otherwise
    """
    is_3d = utils.is3D(smiles)
    return is_3d

@mcp.tool()
@error_handler
@async_timeout(5)
async def example_official_utils(chembl_id: str) -> str:
    """
    Get official name for the ChEMBL ID
    
    Args:
        chembl_id: ChEMBL ID
        
    Returns:
        Official name
    """
    official = utils.official(chembl_id)
    return official

@mcp.tool()
@error_handler
@async_timeout(5)
async def example_removeHs(smiles: str) -> str:
    """
    Remove hydrogen atoms from SMILES string
    
    Args:
        smiles: SMILES string
        
    Returns:
        SMILES string without hydrogen atoms
    """
    smiles_no_h = utils.removeHs(smiles)
    return smiles_no_h

@mcp.tool()
@error_handler
@async_timeout(5)
async def example_smiles2inchi(smiles: str) -> str:
    """
    Convert SMILES string to InChI
    
    Args:
        smiles: SMILES string
        
    Returns:
        InChI string
    """
    smiles_inchi = utils.smiles2inchi(smiles)
    return smiles_inchi

@mcp.tool()
@error_handler
@async_timeout(5)
async def example_smiles2inchiKey(smiles: str) -> str:
    """
    Convert SMILES string to InChI Key
    
    Args:
        smiles: SMILES string
        
    Returns:
        InChI Key
    """
    smiles_inchi_key = utils.smiles2inchiKey(smiles)
    return smiles_inchi_key

@mcp.tool()
@error_handler
@async_timeout(5)
async def example_smiles2svg(smiles: str) -> str:
    """
    Convert SMILES string to SVG image
    
    Args:
        smiles: SMILES string
        
    Returns:
        SVG image string
    """
    smiles_svg = utils.smiles2svg(smiles)
    return smiles_svg
    # print("SMILES SVG:", smiles_svg)  # Skipping printing SVG

@mcp.tool()
@error_handler
@async_timeout(5)
async def example_standardize(smiles: str) -> str:
    """
    Standardize SMILES string
    
    Args:
        smiles: SMILES string
        
    Returns:
        Standardized SMILES string
    """
    standardized_smiles = utils.standardize(smiles)
    return standardized_smiles

@mcp.tool()
@error_handler
@async_timeout(5)
async def example_status() -> Dict[str, Any]:
    """
    Get status information for ChEMBL Web Services
    
    Returns:
        Dictionary of status information
    """
    status = utils.status()
    return status

@mcp.tool()
@error_handler
@async_timeout(5)
async def example_structuralAlerts(smiles: str) -> List[Dict[str, Any]]:
    """
    Get structural alerts for SMILES string
    
    Args:
        smiles: SMILES string
        
    Returns:
        List of structural alerts
    """
    alerts = utils.structuralAlerts(smiles)
    return alerts

if __name__ == "__main__":
    import argparse
    
    # Command line argument parsing
    parser = argparse.ArgumentParser(description='ChEMBL FastMCP Server')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Server host address')
    parser.add_argument('--port', type=int, default=8000, help='Server port')
    parser.add_argument('--transport', type=str, default='http', choices=['http', 'stdio'], help='Transport method')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='Log level')
    
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    logging.info(f"Starting ChEMBL MCP Server (transport: {args.transport})")
    
    # FastMCP only supports 'stdio' as transport parameter
    if args.transport == 'http':
        logging.info(f"HTTP server will run on {args.host}:{args.port}")
        logging.warning("FastMCP does not support http transport, falling back to stdio transport")
        # Fallback to stdio due to missing fastapi module or FastMCP not supporting http
        mcp.run(transport='stdio')
    else:
        logging.info("Using stdio transport")
        mcp.run(transport='stdio')
