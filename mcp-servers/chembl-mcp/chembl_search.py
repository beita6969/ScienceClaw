import chembl_webresource_client
from chembl_webresource_client.new_client import new_client
from chembl_webresource_client.utils import utils
import signal
import time
from functools import wraps

def timeout(seconds):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(f"Function execution exceeded {seconds} seconds")

        @wraps(func)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result
        return wrapper
    return decorator


def example_activity(assay_chembl_id):
    client = new_client
    activities = client.activity.filter(assay_chembl_id=assay_chembl_id)
    return list(activities)


def example_activity_supplementary_data_by_activity(activity_chembl_id):
    client = new_client
    activity_supp_data = client.activity_supplementary_data_by_activity.filter(activity_chembl_id=activity_chembl_id)
    return list(activity_supp_data)


def example_assay(assay_type):
    client = new_client
    assays = client.assay.filter(assay_type=assay_type)
    return list(assays)


def example_assay_class(assay_class_type):
    client = new_client
    assay_classes = client.assay_class.filter(assay_class_type=assay_class_type)
    return list(assay_classes)


def example_atc_class(level1):
    client = new_client
    atc_classes = client.atc_class.filter(level1=level1)
    return list(atc_classes)


def example_binding_site(site_name):
    client = new_client
    binding_sites = client.binding_site.filter(site_name=site_name)
    return list(binding_sites)


def example_biotherapeutic(biotherapeutic_type):
    client = new_client
    biotherapeutics = client.biotherapeutic.filter(biotherapeutic_type=biotherapeutic_type)
    return list(biotherapeutics)


def example_cell_line(cell_line_name):
    client = new_client
    cell_lines = client.cell_line.filter(cell_line_name=cell_line_name)
    return list(cell_lines)


def example_chembl_id_lookup(available_type, q):
    client = new_client
    chembl_ids = client.chembl_id_lookup.filter(available_type=available_type, q=q)
    return list(chembl_ids)


def example_chembl_release():
    client = new_client
    chembl_releases = client.chembl_release.all()
    return list(chembl_releases)


def example_compound_record(compound_name):
    client = new_client
    compound_records = client.compound_record.filter(compound_name=compound_name)
    return list(compound_records)


def example_compound_structural_alert(alert_name):
    client = new_client
    structural_alerts = client.compound_structural_alert.filter(alert_name=alert_name)
    return list(structural_alerts)


def example_description(description_type):
    client = new_client
    descriptions = client.description.filter(description_type=description_type)
    return list(descriptions)


def example_document(journal):
    client = new_client
    documents = client.document.filter(journal=journal)
    return list(documents)


def example_drug(drug_type):
    client = new_client
    drugs = client.drug.filter(drug_type=drug_type)
    return list(drugs)


def example_drug_indication(mesh_heading):
    client = new_client
    drug_indications = client.drug_indication.filter(mesh_heading=mesh_heading)
    return list(drug_indications)


def example_drug_warning(meddra_term):
    client = new_client
    drug_warnings = client.drug_warning.filter(meddra_term=meddra_term)
    return list(drug_warnings)


def example_go_slim(go_slim_term):
    client = new_client
    go_slims = client.go_slim.filter(go_slim_term=go_slim_term)
    return list(go_slims)


def example_mechanism(mechanism_of_action):
    client = new_client
    mechanisms = client.mechanism.filter(mechanism_of_action=mechanism_of_action)
    return list(mechanisms)


def example_molecule(molecule_type):
    client = new_client
    molecules = client.molecule.filter(molecule_type=molecule_type)
    return list(molecules)


def example_molecule_form(form_description):
    client = new_client
    molecule_forms = client.molecule_form.filter(form_description=form_description)
    return list(molecule_forms)


def example_organism(tax_id):
    client = new_client
    organisms = client.organism.filter(tax_id=tax_id)
    return list(organisms)


def example_protein_classification(protein_class_name):
    client = new_client
    protein_classifications = client.protein_classification.filter(protein_class_name=protein_class_name)
    return list(protein_classifications)


def example_source(source_description):
    client = new_client
    sources = client.source.filter(source_description=source_description)
    return list(sources)


def example_target(target_type):
    client = new_client
    targets = client.target.filter(target_type=target_type)
    return list(targets)


def example_target_component(component_type):
    client = new_client
    target_components = client.target_component.filter(component_type=component_type)
    return list(target_components)


def example_target_relation(relationship_type):
    client = new_client
    target_relations = client.target_relation.filter(relationship_type=relationship_type)
    return list(target_relations)


def example_tissue(tissue_name):
    client = new_client
    tissues = client.tissue.filter(tissue_name=tissue_name)
    return list(tissues)


def example_xref_source(xref_name):
    client = new_client
    xref_sources = client.xref_source.filter(xref_name=xref_name)
    return list(xref_sources)


def example_canonicalizeSmiles(smiles):
    canonical_smiles = utils.canonicalizeSmiles(smiles)
    return canonical_smiles


def example_chemblDescriptors(smiles):
    descriptors = utils.chemblDescriptors(smiles)
    return descriptors


def example_description_utils(chembl_id):
    description = utils.description(chembl_id)
    return description


def example_descriptors(smiles):
    descriptors = utils.descriptors(smiles)
    return descriptors


def example_getParent(chembl_id):
    parent = utils.getParent(chembl_id)
    return parent


def example_highlightSmilesFragmentSvg(smiles, fragment):
    highlighted_svg = utils.highlightSmilesFragmentSvg(smiles, fragment)
    return highlighted_svg


def example_inchi2inchiKey(inchi):
    inchi_key = utils.inchi2inchiKey(inchi)
    return inchi_key


def example_inchi2svg(inchi):
    inchi_svg = utils.inchi2svg(inchi)
    return inchi_svg
    # print("InChI SVG:", inchi_svg)  # Skipping printing SVG


def example_is3D(smiles):
    is_3d = utils.is3D(smiles)
    return is_3d


def example_official_utils(chembl_id):
    official = utils.official(chembl_id)
    return official


def example_removeHs(smiles):
    smiles_no_h = utils.removeHs(smiles)
    return smiles_no_h


def example_smiles2inchi(smiles):
    smiles_inchi = utils.smiles2inchi(smiles)
    return smiles_inchi


def example_smiles2inchiKey(smiles):
    smiles_inchi_key = utils.smiles2inchiKey(smiles)
    return smiles_inchi_key


def example_smiles2svg(smiles):
    smiles_svg = utils.smiles2svg(smiles)
    return smiles_svg
    # print("SMILES SVG:", smiles_svg)  # Skipping printing SVG


def example_standardize(smiles):
    standardized_smiles = utils.standardize(smiles)
    return standardized_smiles


def example_status():
    status = utils.status()
    return status


def example_structuralAlerts(smiles):
    alerts = utils.structuralAlerts(smiles)
    return alerts


if __name__ == "__main__":

    # Data entities examples
    def run_with_timeout(func, *args, timeout_seconds=2):
        try:
            start_time = time.time()
            result = func(*args)
            end_time = time.time()
            print(f"Execution time: {end_time - start_time:.2f} seconds")
            return result
        except Exception as e:
            print(f"Execution failed or timed out: {str(e)}")
            return None

    print("\n=== Testing Activity ===")
    assay_chembl_id = 'CHEMBL829585'
    activity_result = run_with_timeout(example_activity, assay_chembl_id)
    if activity_result:
        print("Activity:", activity_result)

    # print("\n=== Testing Activity Supplementary Data ===")
    # activity_chembl_id = 'CHEMBL1172741'
    # activity_supplementary_data_result = run_with_timeout(
    #     example_activity_supplementary_data_by_activity, activity_chembl_id)
    # if activity_supplementary_data_result:
    #     print("Activity Supplementary Data:", activity_supplementary_data_result)

    assay_type = 'B'
    assay_result = example_assay(assay_type)
    print("Assay:", assay_result)

    assay_class_type = 'CELL-BASED'
    assay_class_result = example_assay_class(assay_class_type)
    print("Assay Class:", assay_class_result)

    level1 = 'A'
    atc_class_result = example_atc_class(level1)
    print("ATC Class:", atc_class_result)

    site_name = 'Active Site'
    binding_site_result = example_binding_site(site_name)
    print("Binding Site:", binding_site_result)

    biotherapeutic_type = 'Antibody'
    biotherapeutic_result = example_biotherapeutic(biotherapeutic_type)
    print("Biotherapeutic:", biotherapeutic_result)

    cell_line_name = 'HeLa'
    cell_line_result = example_cell_line(cell_line_name)
    print("Cell Line:", cell_line_result)

    available_type = 'COMPOUND'
    q = 'aspirin'
    chembl_id_lookup_result = example_chembl_id_lookup(available_type, q)
    print("ChEMBL ID Lookup:", chembl_id_lookup_result)

    chembl_release_result = example_chembl_release()
    print("ChEMBL Release:", chembl_release_result)

    compound_name = 'aspirin'
    compound_record_result = example_compound_record(compound_name)
    print("Compound Record:", compound_record_result)

    alert_name = 'Aromatic Nitro'
    compound_structural_alert_result = example_compound_structural_alert(alert_name)
    print("Compound Structural Alert:", compound_structural_alert_result)

    description_type = 'Disease'
    description_result = example_description(description_type)
    print("Description:", description_result)

    journal = 'J. Med. Chem.'
    document_result = example_document(journal)
    print("Document:", document_result)

    drug_type = 'Antibiotic'
    drug_result = example_drug(drug_type)
    print("Drug:", drug_result)

    mesh_heading = 'Hypertension'
    drug_indication_result = example_drug_indication(mesh_heading)
    print("Drug Indication:", drug_indication_result)

    meddra_term = 'Liver injury'
    drug_warning_result = example_drug_warning(meddra_term)
    print("Drug Warning:", drug_warning_result)

    go_slim_term = 'Apoptosis'
    go_slim_result = example_go_slim(go_slim_term)
    print("GO Slim:", go_slim_result)

    mechanism_of_action = 'ACE inhibitor'
    mechanism_result = example_mechanism(mechanism_of_action)
    print("Mechanism:", mechanism_result)

    molecule_type = 'Small molecule'
    molecule_result = example_molecule(molecule_type)
    print("Molecule:", molecule_result)

    form_description = 'Salt'
    molecule_form_result = example_molecule_form(form_description)
    print("Molecule Form:", molecule_form_result)

    tax_id = 9606
    organism_result = example_organism(tax_id)
    print("Organism:", organism_result)

    protein_class_name = 'Kinase'
    protein_classification_result = example_protein_classification(protein_class_name)
    print("Protein Classification:", protein_classification_result)

    source_description = 'ChEMBL'
    source_result = example_source(source_description)
    print("Source:", source_result)

    target_type = 'SINGLE PROTEIN'
    target_result = example_target(target_type)
    print("Target:", target_result)

    component_type = 'PROTEIN'
    target_component_result = example_target_component(component_type)
    print("Target Component:", target_component_result)

    relationship_type = 'SUBUNIT'
    target_relation_result = example_target_relation(relationship_type)
    print("Target Relation:", target_relation_result)

    tissue_name = 'Brain'
    tissue_result = example_tissue(tissue_name)
    print("Tissue:", tissue_result)

    xref_name = 'DrugBank'
    xref_source_result = example_xref_source(xref_name)
    print("XRef Source:", xref_source_result)

    # Utils functions examples
    smiles = 'CC(=O)Oc1ccccc1C(=O)O'
    canonicalize_smiles_result = run_with_timeout(example_canonicalizeSmiles, smiles)
    if canonicalize_smiles_result:
        print("Canonical SMILES:", canonicalize_smiles_result)

    chembl_descriptors_result = example_chemblDescriptors(smiles)
    print("ChEMBL Descriptors:", chembl_descriptors_result)

    chembl_id = 'CHEMBL1'
    description_utils_result = example_description_utils(chembl_id)
    print("Description:", description_utils_result)

    descriptors_result = example_descriptors(smiles)
    print("Descriptors:", descriptors_result)

    parent_result = example_getParent(chembl_id)
    print("Parent:", parent_result)

    fragment = 'c1ccccc1'
    highlight_smiles_fragment_svg_result = example_highlightSmilesFragmentSvg(smiles, fragment)
    # Skipping printing SVG

    inchi = 'InChI=1S/C9H8O4/c1-6(10)13-8-5-3-2-4-7(8)9(11)12/h2-5H,1H3,(H,11,12)'
    inchi_2_inchi_key_result = example_inchi2inchiKey(inchi)
    print("InChI Key:", inchi_2_inchi_key_result)

    inchi_2_svg_result = example_inchi2svg(inchi)
    # Skipping printing SVG

    is_3d_result = example_is3D(smiles)
    print("Is 3D:", is_3d_result)

    official_utils_result = example_official_utils(chembl_id)
    print("Official:", official_utils_result)

    remove_hs_result = example_removeHs(smiles)
    print("SMILES without Hs:", remove_hs_result)

    smiles_2_inchi_result = example_smiles2inchi(smiles)
    print("SMILES InChI:", smiles_2_inchi_result)

    smiles_2_inchi_key_result = example_smiles2inchiKey(smiles)
    print("SMILES InChI Key:", smiles_2_inchi_key_result)

    smiles_2_svg_result = example_smiles2svg(smiles)
    # Skipping printing SVG

    standardize_result = example_standardize(smiles)
    print("Standardized SMILES:", standardize_result)

    status_result = example_status()
    print("Status:", status_result)

    structural_alerts_result = example_structuralAlerts(smiles)
    print("Structural Alerts:", structural_alerts_result)
